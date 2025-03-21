//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --set-elf-op-offsets %s | FileCheck %s
// REQUIRES: arch-NPU40XX

func.func @setOffsets() {
  ELF.Main @ELFMain {
    ELF.CreateLogicalSection @shave.stack aligned(64) secType(SHT_NOBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_SHAVE") {
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_0 : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_1 : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_2 : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_3 : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_4 : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_5 : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_6 : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_7 : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_8 : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_9 : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_10 : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_11 : 16384
    }
    ELF.CreateSection @program.nnrt_config aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") {
      "NPUReg40XX.NNrtConfig"() <{actShaveRt = @shave.runtime::@ActShaveRt, actShaveStacks = [@shave.stack::@ActShaveRtStack_0_0, @shave.stack::@ActShaveRtStack_0_1, @shave.stack::@ActShaveRtStack_0_2, @shave.stack::@ActShaveRtStack_0_3, @shave.stack::@ActShaveRtStack_0_4, @shave.stack::@ActShaveRtStack_0_5, @shave.stack::@ActShaveRtStack_0_6, @shave.stack::@ActShaveRtStack_0_7, @shave.stack::@ActShaveRtStack_0_8, @shave.stack::@ActShaveRtStack_0_9, @shave.stack::@ActShaveRtStack_0_10, @shave.stack::@ActShaveRtStack_0_11], dmaHwpBase = @buffer.CMX_NN.0::@DeclareBuffer6, isActKernelInvocations, sym_name = "MappedInference_nnrtConfigManaged"}> : () -> ()
    }
    ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
      ELF.Symbol @elfsym.shave.stack of(@shave.stack) type(<STT_SECTION>) size(0) value(0)
      ELF.Symbol @elfsym.program.nnrt_config of(@program.nnrt_config) type(<STT_SECTION>) size(0) value(0)
    }
  }
  return
}


//CHECK:   ELF.CreateLogicalSection @shave.stack aligned(64) secType(SHT_NOBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_SHAVE") {
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_0 {elfMemOffsetAttrKey = 0 : ui64} : 16384
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_1 {elfMemOffsetAttrKey = 16384 : ui64} : 16384
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_2 {elfMemOffsetAttrKey = 32768 : ui64} : 16384
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_3 {elfMemOffsetAttrKey = 49152 : ui64} : 16384
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_4 {elfMemOffsetAttrKey = 65536 : ui64} : 16384
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_5 {elfMemOffsetAttrKey = 81920 : ui64} : 16384
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_6 {elfMemOffsetAttrKey = 98304 : ui64} : 16384
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_7 {elfMemOffsetAttrKey = 114688 : ui64} : 16384
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_8 {elfMemOffsetAttrKey = 131072 : ui64} : 16384
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_9 {elfMemOffsetAttrKey = 147456 : ui64} : 16384
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_10 {elfMemOffsetAttrKey = 163840 : ui64} : 16384
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_11 {elfMemOffsetAttrKey = 180224 : ui64} : 16384
