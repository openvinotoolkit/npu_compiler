//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform%" --handle-alignment-requirements --set-elf-op-offsets %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU5020

// Note: This is a hand crafted test case that merges two memory sections
func.func @mergedDDRSections() {
  ELF.Main {
    ELF.CreateLogicalSection @merged.DDR.sections aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_PROC_DMA|VPU_SHF_PROC_SHAVE") secLocation(<DDR>) {
      VPUASM.DeclareBuffer @DeclareBuffer_1 !VPUASM.Buffer< "DDR"[0] <0> : memref<1x1x1x1000xi8, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer_2 !VPUASM.Buffer< "DDR"[0] <1000> : memref<1x1x1x1000xi8, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer_3 !VPUASM.Buffer< "DDR"[0] <0> : memref<1x1x1x1000xi8, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer_4 !VPUASM.Buffer< "DDR"[0] <1000> : memref<1x1x1x1000xi8, @DDR> :  swizzling(0)>

      // CHECK-NOT: ELF.Pad size(96) {elfMemOffsetAttrKey = 4000 : ui64}
      // CHECK:     ELF.Pad size(48) {elfMemOffsetAttrKey = 2000 : ui64}

      // CHECK-NOT: VPUASM.ActShaveRtStack @ActShaveRtStack_0_0_0 {elfMemOffsetAttrKey = 4096 : ui64} : 16384
      // CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_0_0 {elfMemOffsetAttrKey = 2048 : ui64} : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_0_0 : 16384
    }
    ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
      ELF.Symbol @elfsym.merged.DDR.sections of(@merged.DDR.sections) type(<STT_SECTION>) size(0) value(0)
    }
  }
  return
}
