//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-VPUASM-to-NPUReg40XX --create-elf-relocations %s | FileCheck %s
// REQUIRES: arch-NPU40XX
module @OneM2IWithoutAttributes {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x256x256x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x256x256x4xf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x256x256x4xf16, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x256x256x4xf16, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func @main() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @io.NetworkInput0 aligned(1) secType(SHT_PROGBITS) secFlags(VPU_SHF_USERINPUT) {
        VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x256x256x4xf16, @DDR> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @io.NetworkOutput0 aligned(1) secType(SHT_PROGBITS) secFlags(VPU_SHF_USERINPUT) {
        VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x256x256x4xf16, @DDR> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @builtin.tasks.M2I0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_M2I_0 idx(!VPURegMapped.Index<0:0:0>) <M2I>
      }
      ELF.CreateSection @task.m2i0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {
        VPUASM.M2I @M2I_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@builtin.tasks.M2I0::@DeclareTaskBuffer_M2I_0) inputs(@io.NetworkInput0::@DeclareBuffer0) outputs(@io.NetworkOutput0::@DeclareBuffer1) {clean_after = 2 : ui64, do_norm, inFmt = #VPU.m2i_color_fmt<PL_YUV420_8>, norm = [1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01], outFmt = #VPU.m2i_color_fmt<IL_RGB888>, chroma_out_reverse_channels, scale_factor_x = 131072 : ui32, scale_factor_y = 131072 : ui32, start_after = 1 : ui64, updateBarriers = [], waitBarriers = []}
      }
      ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
        ELF.Symbol @elfsym.builtin.tasks.M2I0 of(@builtin.tasks.M2I0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.task.m2i0 of(@task.m2i0) type(<STT_SECTION>) size(0) value(0)
      }
      ELF.CreateSymbolTableSection @symtab.io.NetworkInput secFlags("VPU_SHF_USERINPUT|VPU_SHF_JIT") {
        ELF.Symbol @elfsym.io.NetworkInput0 of(@io.NetworkInput0) type(<STT_SECTION>) size(48) value(0)
      }
      ELF.CreateSymbolTableSection @symtab.io.NetworkOutput secFlags("VPU_SHF_USEROUTPUT|VPU_SHF_JIT") {
        ELF.Symbol @elfsym.io.NetworkOutput0 of(@io.NetworkOutput0) type(<STT_SECTION>) size(48) value(0)
      }

      // CHECK:       ELF.CreateRelocationSection @rela.task.m2i0.symtab.io.NetworkInput
      // CHECK-SAME:    target(@task.m2i0)
      // CHECK-SAME:    symtab(@symtab.io.NetworkInput)
      // CHECK-SAME:    secFlags("VPU_SHF_JIT|VPU_SHF_USERINPUT")
      // CHECK:       ELF.Reloc
      // CHECK-SAME:    offset(0)
      // CHECK-SAME:    sourceSym(@symtab.io.NetworkInput::@elfsym.io.NetworkInput0)
      // CHECK-SAME:    relocType(<R_VPU_64_BIT_OR_B21_B26_UNSET>)
      // CHECK-SAME:    addend(0)
      // CHECK-SAME:    (description : "Input (inputSymRef) in M2I reloc")
      // CHECK:       ELF.Reloc
      // CHECK-SAME:    offset(16)
      // CHECK-SAME:    sourceSym(@symtab.io.NetworkInput::@elfsym.io.NetworkInput0)
      // CHECK-SAME:    relocType(<R_VPU_64_BIT_OR_B21_B26_UNSET>)
      // CHECK-SAME:    addend(43520)
      // CHECK-SAME:    (description : "Input (inAddr1) in M2I reloc")
      // CHECK:       ELF.Reloc
      // CHECK-SAME:    offset(32)
      // CHECK-SAME:    sourceSym(@symtab.io.NetworkInput::@elfsym.io.NetworkInput0)
      // CHECK-SAME:    relocType(<R_VPU_64_BIT_OR_B21_B26_UNSET>)
      // CHECK-SAME:    addend(54400)
      // CHECK-SAME:    (description : "Input (inAddr2) in M2I reloc")

      // CHECK:       ELF.CreateRelocationSection @rela.task.m2i0.symtab.io.NetworkOutput
      // CHECK-SAME:    target(@task.m2i0)
      // CHECK-SAME:    symtab(@symtab.io.NetworkOutput)
      // CHECK-SAME:    secFlags("VPU_SHF_JIT|VPU_SHF_USEROUTPUT")
      // CHECK:       ELF.Reloc
      // CHECK-SAME:    offset(96)
      // CHECK-SAME:    sourceSym(@symtab.io.NetworkOutput::@elfsym.io.NetworkOutput0)
      // CHECK-SAME:    relocType(<R_VPU_64_BIT_OR_B21_B26_UNSET>)
      // CHECK-SAME:    addend(0)
      // CHECK-SAME:    (description : "Output (outputSymRef) in M2I reloc")
    }
    return
  }
}
