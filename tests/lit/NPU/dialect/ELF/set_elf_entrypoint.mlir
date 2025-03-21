//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --set-elf-entrypoint %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
IE.CNNNetwork entryPoint : @oneDma inputsInfo : {
  DataInfo "input" : tensor<1x2x3x4xf16>
} outputsInfo : {
  DataInfo "output" : tensor<1x2x3x4xf16>
}
func.func @oneDma() {
  ELF.Main @ELFMain {
    ELF.CreateSection @note.MappedInferenceVersion aligned(4) secType(SHT_NOTE) secFlags("SHF_NONE") {
      VPUASM.MappedInferenceVersion @MappedInferenceVersion_0_0(11 _ 4 _ 10)
    }
    ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
      VPUASM.MappedInference @MappedInference : dmaCount([[0, 0]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([0]) actKernelInvocationsCount([0]) mediaCount(0) barrierCount(0) mappedInferenceVersion(@note.MappedInferenceVersion::@MappedInferenceVersion_0_0)
    }
    ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
      ELF.Symbol @elfsym.program.mapped_inference of(@program.mapped_inference) type(<STT_SECTION>) size(0) value(0)
    }
  }
  return
}

//CHECK: ELF.Symbol @entry of(@program.mapped_inference::@MappedInference) type(<VPU_STT_ENTRY>) size(0) value(0)
