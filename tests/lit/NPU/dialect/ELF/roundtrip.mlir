//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: env IE_NPU_LOG_FILTER=LOG_ERROR vpux-opt --init-compiler="vpu-arch=%arch%" %s | vpux-opt --init-compiler="vpu-arch=%arch% allow-custom-values=true" | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

func.func @elf_roundtrip() {
  ELF.Main @ELFMain {
    ELF.CreateSection @data1 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) secLocation(<DDR>) {
    }
    ELF.CreateLogicalSection @buffer1 aligned(64) secType(SHT_NOBITS) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
    }
    ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
    ELF.Symbol @sym_data1 of(@data1) type(<STT_SECTION>) size(0) value(0)
    ELF.Symbol @sym_buffer1 of(@buffer1) type(<STT_SECTION>) size(0) value(0)
    }
    ELF.CreateSymbolTableSection @inputs secFlags(VPU_SHF_USERINPUT) {
    }
    ELF.CreateSymbolTableSection @outputs secFlags(VPU_SHF_USERINPUT) {
    }
    ELF.CreateRelocationSection @relocs target(@data1) symtab(@symtab) secFlags("SHF_NONE") {
    }
    ELF.CreateRelocationSection @inputRelocs target(@data1) symtab(@inputs) secFlags(VPU_SHF_USERINPUT) {
    }
  }
  return
}

//CHECK: ELF.Main @ELFMain {
//CHECK: ELF.CreateSection @data1 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) secLocation(<DDR>)
//CHECK: ELF.CreateLogicalSection @buffer1 aligned(64) secType(SHT_NOBITS) secFlags("SHF_NONE") secLocation(<CMX_NN>)
//CHECK: ELF.CreateSymbolTableSection @symtab  secFlags("SHF_NONE")
//CHECK: ELF.Symbol @sym_data1 of(@data1) type(<STT_SECTION>) size(0) value(0)
//CHECK: ELF.Symbol @sym_buffer1 of(@buffer1) type(<STT_SECTION>) size(0) value(0)
//CHECK: ELF.CreateSymbolTableSection @inputs  secFlags(VPU_SHF_USERINPUT)
//CHECK: ELF.CreateSymbolTableSection @outputs  secFlags(VPU_SHF_USERINPUT)
//CHECK: ELF.CreateRelocationSection @relocs target(@data1) symtab(@symtab) secFlags("SHF_NONE")
//CHECK: ELF.CreateRelocationSection @inputRelocs target(@data1) symtab(@inputs) secFlags(VPU_SHF_USERINPUT)
