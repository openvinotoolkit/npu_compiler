//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --cmx-stack-frames-reserve-mem --cmx-metadata-reserve-mem --dma-task-profiling-reserve-mem --lower-VPUIP-to-ELF %s | FileCheck %s
// REQUIRES: platform-NPU4000

module @OneDMAWithoutAttributes {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <1571328> -> memref<1x2x3x4xf16, [@CMX_NN, 0]>
    %1 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %2 = VPURT.ConfigureBarrier<1> <{isFinalBarrier}> -> !VPURT.Barrier
    VPURT.Task updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%0 : memref<1x2x3x4xf16, [@CMX_NN, 0]>) -> memref<1x2x3x4xf16, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%1 : !VPURT.Barrier) updates(%2 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%0 : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    return %arg1 : memref<1x2x3x4xf16, @DDR>
  }
  // CHECK:       ELF.Main
  // CHECK-DAG:       ELF.CreateLogicalSection @io.NetworkInput.0
  // CHECK-DAG:       ELF.CreateLogicalSection @io.NetworkOutput.0

  // CHECK:       ELF.CreateSection @task.dma.0.0
  // CHECK:       ELF.CreateSection @task.dma.0.1
  // CHECK:       NPUReg40XX.NNDMA

  // CHECK:       ELF.CreateSection @program.mapped_inference
  // CHECK:       NPUReg40XX.MappedInference

  // CHECK:       ELF.CreateSymbolTableSection @symtab.io.NetworkInput
  // CHECK:       ELF.Symbol @elfsym.io.NetworkInput.0 of(@io.NetworkInput.0)

  // CHECK:       ELF.CreateSymbolTableSection @symtab.io.NetworkOutput
  // CHECK:       ELF.Symbol @elfsym.io.NetworkOutput.0 of(@io.NetworkOutput.0)

  // CHECK:       ELF.CreateSymbolTableSection @symtab
  // CHECK:       ELF.Symbol @elfsym.program.managedBarrier of(@program.managedBarrier)
  // CHECK:       ELF.Symbol @elfsym.program.bootstrap of(@program.bootstrap)
  // CHECK:       ELF.Symbol @elfsym.program.workItem of(@program.workItem)
  // CHECK:       ELF.Symbol @elfsym.program.mapped_inference of(@program.mapped_inference)
  // CHECK:       ELF.Symbol @entry of(@program.mapped_inference::@MappedInference)

  // CHECK:       ELF.CreateMetadataSection @MetadataSection
  // CHECK:       VPUASM.NetworkMetadata @NetworkMetadata

  // CHECK:       ELF.CreateRelocationSection
  // CHECK-SAME:  NetworkInput
  // CHECK-SAME:  VPU_SHF_JIT|VPU_SHF_USERINPUT
  // CHECK:       ELF.Reloc
}
