//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true workload-management-enable=false" --lower-VPUIP-to-ELF="workload-management-enable=false" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @OneDMAWithoutAttributes {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  config.Resources 6 of @NCE at 1.700000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        config.MemoryResource 512 bytes of @CMX_NN offset 0
      }
    }
  }
  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x3x4xf16, [@CMX_NN, 0]>
    %1 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    VPURT.Task updates(%1 : !VPURT.Barrier) {
      %2 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%0 : memref<1x2x3x4xf16, [@CMX_NN, 0]>) -> memref<1x2x3x4xf16, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%1 : !VPURT.Barrier) {
      %2 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    return %arg1 : memref<1x2x3x4xf16, @DDR>
  }
  // CHECK:       ELF.Main @ELFMain
  // CHECK-DAG:       ELF.CreateLogicalSection @io.NetworkInput.0
  // CHECK-DAG:       ELF.CreateLogicalSection @io.NetworkOutput.0
  // CHECK-DAG:       ELF.CreateLogicalSection @program.metadata.cmx
  // CHECK-DAG:       ELF.CreateLogicalSection @buffer.CMX_NN.0

  // CHECK:       ELF.CreateSection @program.barrier
  // CHECK:       NPUReg40XX.ConfigureBarrier

  // CHECK:       ELF.CreateSection @task.dma.0.0
  // CHECK:       NPUReg40XX.NNDMA
  // CHECK:            dma_dst_addr {
  // CHECK:                UINT dma_dst = 0x40218000,

  // CHECK:       ELF.CreateSection @task.dma.0.1
  // CHECK:       NPUReg40XX.NNDMA
  // CHECK:          dma_src_addr {
  // CHECK:            UINT dma_src = 0x40218000,

  // CHECK:       ELF.CreateSection @program.mapped_inference
  // CHECK:       NPUReg40XX.MappedInference
  // CHECK:          miLogAddrDmaHwp = UINT 0x40218000,

  // CHECK:       ELF.CreateSymbolTableSection @symtab
  // CHECK:       ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx)
  // CHECK:       ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0)
  // CHECK:       ELF.Symbol @elfsym.program.barrier of(@program.barrier)
  // CHECK:       ELF.Symbol @elfsym.task.dma.0.0 of(@task.dma.0.0)
  // CHECK:       ELF.Symbol @elfsym.task.dma.0.1 of(@task.dma.0.1)
  // CHECK:       ELF.Symbol @elfsym.program.mapped_inference of(@program.mapped_inference)
  // CHECK:       ELF.Symbol @entry of(@program.mapped_inference::@MappedInference)

  // CHECK:       ELF.CreateSymbolTableSection @symtab.io.NetworkInput
  // CHECK:       ELF.Symbol @elfsym.io.NetworkInput.0 of(@io.NetworkInput.0)

  // CHECK:       ELF.CreateSymbolTableSection @symtab.io.NetworkOutput
  // CHECK:       ELF.Symbol @elfsym.io.NetworkOutput.0 of(@io.NetworkOutput.0)

  // CHECK:       ELF.CreateMetadataSection @MetadataSection
  // CHECK:       VPUASM.NetworkMetadata @NetworkMetadata

  // CHECK:       ELF.CreateRelocationSection
  // CHECK-SAME:  NetworkInput
  // CHECK-SAME:  VPU_SHF_JIT|VPU_SHF_USERINPUT
  // CHECK:       ELF.Reloc
  // CHECK-SAME:  NetworkInput

  // CHECK-NOT:      ELF.CreateRelocationSection @rela.task.dma.0.0.symtab target(@task.dma.0.0)

  // CHECK-NOT:      ELF.CreateRelocationSection @rela.task.dma.0.1.symtab target(@task.dma.0.1)

  // CHECK:       ELF.CreateRelocationSection
  // CHECK-SAME:  NetworkOutput
  // CHECK-SAME:  VPU_SHF_JIT|VPU_SHF_USEROUTPUT
  // CHECK:       ELF.Reloc

  // CHECK:       ELF.CreateRelocationSection
  // CHECK-SAME:  mapped_inference
  // CHECK:       ELF.Reloc
  // CHECK:       ELF.Reloc
  // CHECK:       ELF.Reloc
  // CHECK-NOT:   ELF.Reloc offset(16) sourceSym(@symtab::@elfsym.buffer.CMX_NN.0)
}
