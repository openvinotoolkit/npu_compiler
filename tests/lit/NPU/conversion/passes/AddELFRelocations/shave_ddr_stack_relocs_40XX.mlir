//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --create-elf-relocations %s | FileCheck %s
// REQUIRES: arch-NPU40XX

func.func @main() {
ELF.Main @ELFMain {
    ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") {
      VPUASM.DeclareBuffer @DeclareBuffer6 !VPUASM.Buffer< "CMX_NN"[0] <1473536> : memref<16xui32, [@CMX_NN, 0]> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer7 !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x1x1xf16, [@CMX_NN, 0]> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer8 !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x64x1x1xf16, [@CMX_NN, 0]> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer9 !VPUASM.Buffer< "CMX_NN"[0] <256> : memref<1x64x1x1xf32, [@CMX_NN, 0]> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer10 !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x32x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [64, 1, 1, 1]}, [@CMX_NN, 0]> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer11 !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x32x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [64, 1, 1, 1]}, [@CMX_NN, 0]> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer12 !VPUASM.Buffer< "CMX_NN"[0] <64> : memref<1x32x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [64, 1, 1, 1]}, [@CMX_NN, 0]> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer13 !VPUASM.Buffer< "CMX_NN"[0] <192> : memref<1x32x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [64, 1, 1, 1]}, [@CMX_NN, 0]> :  swizzling(0)>
    }
    ELF.CreateSection @shave.runtime aligned(1024) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_SHAVE") {
      "NPUReg40XX.ActShaveRt"() <{kernel_path = "nnActEntry", sym_name = "ActShaveRt"}> {elfMemOffsetAttrKey = 0 : ui64} : () -> ()
    }
    ELF.CreateLogicalSection @shave.stack aligned(64) secType(SHT_NOBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_SHAVE") {
    VPUASM.ActShaveRtStack @ActShaveRtStack_0_0 {elfMemOffsetAttrKey = 0 : ui64} : 16384
    VPUASM.ActShaveRtStack @ActShaveRtStack_0_1 {elfMemOffsetAttrKey = 16384 : ui64} : 16384
    VPUASM.ActShaveRtStack @ActShaveRtStack_0_2 {elfMemOffsetAttrKey = 32768 : ui64} : 16384
    VPUASM.ActShaveRtStack @ActShaveRtStack_0_3 {elfMemOffsetAttrKey = 49152 : ui64} : 16384
    VPUASM.ActShaveRtStack @ActShaveRtStack_0_4 {elfMemOffsetAttrKey = 65536 : ui64} : 16384
    VPUASM.ActShaveRtStack @ActShaveRtStack_0_5 {elfMemOffsetAttrKey = 81920 : ui64} : 16384
    VPUASM.ActShaveRtStack @ActShaveRtStack_0_6 {elfMemOffsetAttrKey = 98304 : ui64} : 16384
    VPUASM.ActShaveRtStack @ActShaveRtStack_0_7 {elfMemOffsetAttrKey = 114688 : ui64} : 16384
    VPUASM.ActShaveRtStack @ActShaveRtStack_0_8 {elfMemOffsetAttrKey = 131072 : ui64} : 16384
    VPUASM.ActShaveRtStack @ActShaveRtStack_0_9 {elfMemOffsetAttrKey = 147456 : ui64} : 16384
    VPUASM.ActShaveRtStack @ActShaveRtStack_0_10 {elfMemOffsetAttrKey = 163840 : ui64} : 16384
    VPUASM.ActShaveRtStack @ActShaveRtStack_0_11 {elfMemOffsetAttrKey = 180224 : ui64} : 16384
    }
    ELF.CreateSection @program.nnrt_config aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") {
    "NPUReg40XX.NNrtConfig"() <{actShaveRt = @shave.runtime::@ActShaveRt, actShaveStacks = [@shave.stack::@ActShaveRtStack_0_0, @shave.stack::@ActShaveRtStack_0_1, @shave.stack::@ActShaveRtStack_0_2, @shave.stack::@ActShaveRtStack_0_3, @shave.stack::@ActShaveRtStack_0_4, @shave.stack::@ActShaveRtStack_0_5, @shave.stack::@ActShaveRtStack_0_6, @shave.stack::@ActShaveRtStack_0_7, @shave.stack::@ActShaveRtStack_0_8, @shave.stack::@ActShaveRtStack_0_9, @shave.stack::@ActShaveRtStack_0_10, @shave.stack::@ActShaveRtStack_0_11], dmaHwpBase = @buffer.CMX_NN.0::@DeclareBuffer6, isActKernelInvocations, sym_name = "MappedInference_nnrtConfigManaged"}> {elfMemOffsetAttrKey = 0 : ui64} : () -> ()
    }
    ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
    ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>) size(0) value(0)
    ELF.Symbol @elfsym.shave.runtime of(@shave.runtime) type(<STT_SECTION>) size(0) value(0)
    ELF.Symbol @elfsym.shave.stack of(@shave.stack) type(<STT_SECTION>) size(0) value(0)
    ELF.Symbol @elfsym.program.nnrt_config of(@program.nnrt_config) type(<STT_SECTION>) size(0) value(0)
    }
}
return
}


// CHECK:  ELF.CreateRelocationSection @rela.program.nnrt_config.symtab target(@program.nnrt_config) symtab(@symtab) secFlags("SHF_NONE") {
// CHECK:         ELF.Reloc offset(16) sourceSym(@symtab::@elfsym.shave.runtime) relocType(<R_VPU_64>) addend(0) (description : "actShaveRt in mapped inference reloc")
// CHECK:         ELF.Reloc offset(24) sourceSym(@symtab::@elfsym.shave.stack) relocType(<R_VPU_32>) addend(16384) (description : "Act shave stack in mapped inference reloc")
// CHECK:         ELF.Reloc offset(28) sourceSym(@symtab::@elfsym.shave.stack) relocType(<R_VPU_32>) addend(32768) (description : "Act shave stack in mapped inference reloc")
// CHECK:         ELF.Reloc offset(32) sourceSym(@symtab::@elfsym.shave.stack) relocType(<R_VPU_32>) addend(49152) (description : "Act shave stack in mapped inference reloc")
// CHECK:         ELF.Reloc offset(36) sourceSym(@symtab::@elfsym.shave.stack) relocType(<R_VPU_32>) addend(65536) (description : "Act shave stack in mapped inference reloc")
// CHECK:         ELF.Reloc offset(40) sourceSym(@symtab::@elfsym.shave.stack) relocType(<R_VPU_32>) addend(81920) (description : "Act shave stack in mapped inference reloc")
// CHECK:         ELF.Reloc offset(44) sourceSym(@symtab::@elfsym.shave.stack) relocType(<R_VPU_32>) addend(98304) (description : "Act shave stack in mapped inference reloc")
// CHECK:         ELF.Reloc offset(48) sourceSym(@symtab::@elfsym.shave.stack) relocType(<R_VPU_32>) addend(114688) (description : "Act shave stack in mapped inference reloc")
// CHECK:         ELF.Reloc offset(52) sourceSym(@symtab::@elfsym.shave.stack) relocType(<R_VPU_32>) addend(131072) (description : "Act shave stack in mapped inference reloc")
// CHECK:         ELF.Reloc offset(56) sourceSym(@symtab::@elfsym.shave.stack) relocType(<R_VPU_32>) addend(147456) (description : "Act shave stack in mapped inference reloc")
// CHECK:         ELF.Reloc offset(60) sourceSym(@symtab::@elfsym.shave.stack) relocType(<R_VPU_32>) addend(163840) (description : "Act shave stack in mapped inference reloc")
// CHECK:         ELF.Reloc offset(64) sourceSym(@symtab::@elfsym.shave.stack) relocType(<R_VPU_32>) addend(180224) (description : "Act shave stack in mapped inference reloc")
// CHECK:         ELF.Reloc offset(68) sourceSym(@symtab::@elfsym.shave.stack) relocType(<R_VPU_32>) addend(196608) (description : "Act shave stack in mapped inference reloc")
// CHECK:         ELF.Reloc offset(96) sourceSym(@symtab::@elfsym.buffer.CMX_NN.0) relocType(<R_VPU_64>) addend(1473536) (description : "")
