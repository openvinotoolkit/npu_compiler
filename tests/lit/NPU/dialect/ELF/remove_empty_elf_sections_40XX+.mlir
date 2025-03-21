//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --remove-empty-ELF-sections %s | FileCheck %s
// REQUIRES: arch-NPU40XX
func.func @oneDma() {
  ELF.Main @ELFMain {
    ELF.CreateLogicalSection @buffer.CMX_NN aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") {
      VPUASM.DeclareBuffer @DeclareBuffer !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x10x2x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]> :  swizzling(0)>
    }
    ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") {
      VPUASM.DeclareTaskBuffer @DeclareTaskBuffer idx(!VPURegMapped.Index<0:0:0>) <ActKernelInvocation> {elfMemOffsetAttrKey = 53760 : ui64}
    }
    ELF.CreateLogicalSection @buffer.empty aligned(64) secType(SHT_NOBITS) secFlags("SHF_NONE") {
    }
    VPUASM.DeclareKernelEntry @DeclareKernelEntry : "activation_sigmoid"
    ELF.CreateSection @shave.text aligned(1024) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
      VPUASM.DeclareKernelText @DeclareKernelText {elfMemOffsetAttrKey = 0 : ui64} : "activation_sigmoid"
    }
    ELF.CreateSection @data.empty aligned(64) secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {
    }
    ELF.CreateSymbolTableSection @symtab.0 secFlags("SHF_NONE") {
      ELF.Symbol @elfsym.buffer.empty of(@buffer.empty) type(<STT_SECTION>) size(0) value(0)
      ELF.Symbol @elfsym.data.empty of(@data.empty) type(<STT_SECTION>) size(0) value(0)
    }
    ELF.CreateSymbolTableSection @symtab.1 secFlags("SHF_NONE") {
      ELF.Symbol @elfsym.buffer.CMX_NN of(@buffer.CMX_NN) type(<STT_SECTION>) size(0) value(0)
      ELF.Symbol @elfsym.data.empty of(@data.empty) type(<STT_SECTION>) size(0) value(0)
    }
  }
  return

    // CHECK-LABEL: @oneDma
    // CHECK:       ELF.CreateLogicalSection @buffer.CMX_NN
    // CHECK-NEXT:  VPUASM.DeclareBuffer @DeclareBuffer
    // CHECK:       ELF.CreateLogicalSection @program.metadata.cmx
    // CHECK-NEXT:  VPUASM.DeclareTaskBuffer @DeclareTaskBuffer
    // CHECK-NOT:   ELF.CreateLogicalSection @buffer.empty
    // CHECK:       VPUASM.DeclareKernelEntry @DeclareKernelEntry
    // CHECK:       ELF.CreateSection @shave.text
    // CHECK-NEXT:  VPUASM.DeclareKernelText @DeclareKernelText
    // CHECK-NOT:   ELF.CreateSection @data.empty
    // CHECK-NOT:   ELF.CreateSymbolTableSection @symtab.0
    // CHECK-NOT:   ELF.Symbol @elfsym.buffer.empty
    // CHECK-NOT:   ELF.Symbol @elfsym.data.empty
    // CHECK:       ELF.CreateSymbolTableSection @symtab.1
    // CHECK-NEXT:  ELF.Symbol @elfsym.buffer.CMX_NN
    // CHECK-NOT:   ELF.Symbol @elfsym.data.empty

}
