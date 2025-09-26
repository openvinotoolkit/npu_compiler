//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --add-compiler-hash %s | FileCheck %s
// REQUIRES: arch-NPU40XX

net.NetworkInfo entryPoint : @oneDma inputsInfo : {
  DataInfo "input" : tensor<1x2x3x4xf16>
} outputsInfo : {
  DataInfo "output" : tensor<1x2x3x4xf16>
}
func.func @oneDma() {
  ELF.Main @ELFMain {
    ELF.CreateSection @note.MappedInferenceVersion aligned(4) secType(SHT_NOTE) secFlags("SHF_NONE") secLocation(<DDR>) {
      VPUASM.MappedInferenceVersion @MappedInferenceVersion_0_0(11 _ 4 _ 10)
    }

    ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
      VPUASM.MappedInference @MappedInference : dmaCount([[0, 0]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([0]) actKernelInvocationsCount([0]) mediaCount(0) barrierCount(0) mappedInferenceVersion(@note.MappedInferenceVersion::@MappedInferenceVersion_0_0)
    }
  }
  return
}

// CHECK: ELF.CompilerHash
