//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//


// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --convert-VPUASM-to-NPUReg40XX --create-elf-relocations %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @Model20 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Input" : tensor<1x50x1x1xf16>
  } outputsInfo : {
    DataInfo "Sigmoid_225" : tensor<1x50x1x1xf16>
  }

  func.func @main() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<DDR>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_ActKernelRange_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <ActKernelRange> {elfMemOffsetAttrKey = 51200 : ui64}
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_ActKernelInvocation_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <ActKernelInvocation> {elfMemOffsetAttrKey = 53760 : ui64}
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer5 !VPUASM.Buffer< "CMX_NN"[0] <120> : memref<1x10x2x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer6 !VPUASM.Buffer< "CMX_NN"[0] <240> : memref<1x10x2x3xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]> :  swizzling(0)>
      }
      VPUASM.DeclareKernelEntry @DeclareKernelEntry_0_0 : "activation_sigmoid"

      ELF.CreateSection @shave.text aligned(1024) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DeclareKernelText @DeclareKernelText_0_0 {elfMemOffsetAttrKey = 0 : ui64} : "activation_sigmoid"
      }
      ELF.CreateSection @shave.data aligned(1024) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DeclareKernelData @DeclareKernelArgs_0_0 {elfMemOffsetAttrKey = 0 : ui64} : "activation_sigmoid"
      }
      ELF.CreateSection @shave.params aligned(1024) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.KernelParams @KernelParams_0_0 inputs([@buffer.CMX_NN.0::@DeclareBuffer5]) outputs([@buffer.CMX_NN.0::@DeclareBuffer6]) dynamicInputShapes([]) dynamicOutputShapes([]) kernel_type("activation_sigmoid") {elfMemOffsetAttrKey = 0 : ui64} < { kernel_params = [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]}>
      }
      ELF.CreateSection @task.shave.range.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.ActKernelRange @ActKernelRange_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_ActKernelRange_0_0_0) kernelTaskType(@COMPUTE) calls @shave.text::@DeclareKernelText_0_0 : @DeclareKernelEntry_0_0 {elfMemOffsetAttrKey = 0 : ui64}
      }
      ELF.CreateSection @task.shave.invocation.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.ActKernelInvocation @ActKernelInvocation_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_ActKernelInvocation_0_0_0) -> @program.metadata.cmx::@DeclareTaskBuffer_ActKernelRange_0_0_0(kernel_data : @shave.data::@DeclareKernelArgs_0_0, kernel_params : @shave.params::@KernelParams_0_0) waits([0 : ui8]) updates([1 : ui8]) tile(0) start_after(2) clean_after(1) range_index(0) {elfMemOffsetAttrKey = 0 : ui64}
      }
      ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
        ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>) size(82944) value(1075854336)
        ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>) size(1474560) value(1075937280)
        ELF.Symbol @elfsym.shave.text of(@shave.text) type(<STT_SECTION>)
        ELF.Symbol @elfsym.shave.data of(@shave.data) type(<STT_SECTION>)
        ELF.Symbol @elfsym.shave.params of(@shave.params) type(<STT_SECTION>)
        ELF.Symbol @elfsym.task.shave.range.0.0 of(@task.shave.range.0.0) type(<STT_SECTION>)
        ELF.Symbol @elfsym.task.shave.invocation.0.0 of(@task.shave.invocation.0.0) type(<STT_SECTION>)
      }

    }
    return
}

    // CHECK: ELF.CreateSection @shave.params aligned(1024) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
    // CHECK:    VPUASM.KernelParams
    // CHECK: {elfMemOffsetAttrKey = 0 : ui64}
    // CHECK: < {kernel_params = [120, 128, 1, 64, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 240, 128, 1, 64, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]}>

    // CHECK:    VpuActKernelInvocation {
    // CHECK:        range = UINT 0x40203C00,
    // CHECK:      ELF.CreateRelocationSection @rela.shave.params.symtab
    // CHECK-SAME:   target(@shave.params)
    // CHECK-SAME:   symtab(@symtab)
    // CHECK-SAME:   secFlags("SHF_NONE")
    // CHECK:      ELF.Reloc
    // CHECK-SAME:   offset({{[0-9]+}})
    // CHECK-SAME:   sourceSym(@symtab::@elfsym.shave.params)
    // CHECK-SAME:   relocType(<R_VPU_32>)
    // CHECK-SAME:   addend({{[0-9]+}})
    // CHECK-SAME:   (description : "Input 0 dims (dimsAddr) kernel params reloc")
    // CHECK:      ELF.Reloc
    // CHECK-SAME:   offset({{[0-9]+}})
    // CHECK-SAME:   sourceSym(@symtab::@elfsym.shave.params)
    // CHECK-SAME:   relocType(<R_VPU_32>)
    // CHECK-SAME:   addend({{[0-9]+}})
    // CHECK-SAME:   (description : "Input 0 strides (stridesAddr) kernel params reloc")
    // CHECK:      ELF.Reloc
    // CHECK-SAME:   offset({{[0-9]+}})
    // CHECK-SAME:   sourceSym(@symtab::@elfsym.shave.params)
    // CHECK-SAME:   relocType(<R_VPU_32>)
    // CHECK-SAME:   addend({{[0-9]+}})
    // CHECK-SAME:   (description : "Output 0 dims (dimsAddr) kernel params reloc")
    // CHECK:      ELF.Reloc
    // CHECK-SAME:   offset({{[0-9]+}})
    // CHECK-SAME:   sourceSym(@symtab::@elfsym.shave.params)
    // CHECK-SAME:   relocType(<R_VPU_32>)
    // CHECK-SAME:   addend({{[0-9]+}})
    // CHECK-SAME:   (description : "Output 0 strides (stridesAddr) kernel params reloc")

    // CHECK:      ELF.CreateRelocationSection @rela.task.shave.range.0.0.symtab
    // CHECK-SAME:   target(@task.shave.range.0.0)
    // CHECK-SAME:   symtab(@symtab)
    // CHECK-SAME:   secFlags("SHF_NONE")
    // CHECK:      ELF.Reloc
    // CHECK-SAME:   offset({{[0-9]+}})
    // CHECK-SAME:   sourceSym(@symtab::@elfsym.shave.text)
    // CHECK-SAME:   relocType(<R_VPU_64>)
    // CHECK-SAME:   addend({{[0-9]+}})
    // CHECK-SAME:   (description : "Kernel text (ptr in text_window_base) for act kernel range reloc")

    // CHECK:      ELF.CreateRelocationSection @rela.task.shave.invocation.0.0.symtab
    // CHECK-SAME:   target(@task.shave.invocation.0.0)
    // CHECK-SAME:   symtab(@symtab)
    // CHECK-SAME:   secFlags("SHF_NONE")

    // CHECK-NOT:  ELF.Reloc offset({{[0-9]+}}) sourceSym(@symtab::@elfsym.program.metadata.cmx)

    // CHECK:      ELF.Reloc
    // CHECK-SAME:   offset({{[0-9]+}})
    // CHECK-SAME:   sourceSym(@symtab::@elfsym.shave.data)
    // CHECK-SAME:   relocType(<R_VPU_64>)
    // CHECK-SAME:   addend({{[0-9]+}})
    // CHECK-SAME:   (description : "Kernel data in act kernel invocation reloc")
    // CHECK:      ELF.Reloc
    // CHECK-SAME:   offset({{[0-9]+}})
    // CHECK-SAME:   sourceSym(@symtab::@elfsym.shave.params)
    // CHECK-SAME:   relocType(<R_VPU_64>)
    // CHECK-SAME:   addend({{[0-9]+}})
    // CHECK-SAME:   (description : "Kernel params in act kernel invocation reloc")
}
