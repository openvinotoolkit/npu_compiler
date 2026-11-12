//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform% allow-custom-values=true" --cmx-stack-frames-reserve-mem --cmx-metadata-reserve-mem --split-input-file --convert-VPUASM-to-NPUReg60XX --create-elf-relocations %s | FileCheck %s
// REQUIRES: platform-NPU6010

module @HostParsedInferenceReloc {
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
  func.func @main() {
    ELF.Main {
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
        VPUASM.ManagedMappedInference @MappedInference_managed : dmas ([]) workItems() barrierTasks() bootstrapBarriers()
         nnrtConfig(@program.nnrt_config::@MappedInference_nnrtConfigManaged)
         mappedInferenceVersion(@note.MappedInferenceVersion::@MappedInferenceVersion_0_0) dpuPvpTask(@dpu.pvp::@DpuPVP_0_0)
         {actshv_used = 0 : ui8, barrierConfigurationStride = 0 : i64, barrierConfigurationTasksCount = 0 : i64, barrierCount = 0 : i64,
          barriersReprogrammingCount = 0 : i64, bootstrapWorkItemsCount = 0 : i64, bootstrapBarriersCount = 0 : i64, disableDmaSwFifo = true,
          dmaCount = [[0, 0], [0, 0]], dma_from_cmx_used = 0 : ui8, dma_from_ddr_used = 0 : ui8, dpu_used = 1 : ui8, final_barrier_id = 0 : i64,
          workItemsCount = 0 : i64,
          workloadManagementBarrierProgrammingMode = #VPURegMapped.workload_management_barrier_programming_mode<ALL_BARRIER_DMAS_SCHEDULED>}
      }
      ELF.CreateSection @program.host_parsed_inference aligned(64) secType(VPU_SHT_HPI) secFlags("SHF_ALLOC") secLocation(<DDR>) {
        VPUASM.HostParsedInference @HostParsedInference : mapped(@program.mapped_inference::@MappedInference_managed) nnSliceLength(256) nnSliceCount(2) nnBarriers(0)
      }
      ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
        ELF.Symbol @elfsym.dpu.pvp of(@dpu.pvp) type(<STT_SECTION>)
        ELF.Symbol @elfsym.program.nnrt_config of(@program.nnrt_config) type(<STT_SECTION>)
        ELF.Symbol @elfsym.program.mapped_inference of(@program.mapped_inference) type(<STT_SECTION>)
        ELF.Symbol @elfsym.program.host_parsed_inference of(@program.host_parsed_inference) type(<STT_SECTION>)
        ELF.Symbol @entry of(@program.mapped_inference::@MappedInference_managed) type(<VPU_STT_ENTRY>)
      }
    }
    return
  }
}

// CHECK:      ELF.CreateRelocationSection @rela.program.host_parsed_inference.symtab
// CHECK-SAME:   target(@program.host_parsed_inference)
// CHECK-SAME:   symtab(@symtab)
// mapped_ VpuTaskReference address relocation points to the ManagedMappedInference section
// CHECK:      ELF.Reloc
// CHECK-SAME:   offset(368)
// CHECK-SAME:   sourceSym(@symtab::@elfsym.program.mapped_inference)
// CHECK-SAME:   relocType(<R_VPU_64>)
