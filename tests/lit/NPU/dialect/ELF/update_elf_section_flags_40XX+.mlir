//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//


// RUN: vpux-opt --vpu-arch=%arch% --update-ELF-section-flags %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @mainModule attributes {config.arch = #config.arch_kind<NPU40XX>} {

  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @cache_flush_invalidate() attributes {VPU.task_type = @CACHE_FLUSH_INVALIDATE}
    func.func private @builtin_Minimum(memref<*xf16>, memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "eltwise_min.cpp", VPU.kernel_entry = "eltwise_min", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

  config.Resources 1 of @NCE at 1.700000e+03 MHz
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter_224" : tensor<64x32x32x16xf16>
  } outputsInfo : {
    DataInfo "Minimum_226" : tensor<64x32x32x16xf16>
  }

  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @Parameter_224_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<64x32x32x16xf16, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @Minimum_226_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<64x32x32x16xf16, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }

func.func @main() {
  ELF.Main @ELFMain {
    // CHECK:    ELF.CreateLogicalSection @io.NetworkInput0
    // CHECK-SAME:    secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USERINPUT|VPU_SHF_PROC_SHAVE")
    ELF.CreateLogicalSection @io.NetworkInput0 aligned(1) secType(SHT_NOBITS) secFlags("VPU_SHF_USERINPUT|SHF_WRITE|SHF_ALLOC") secLocation(<NetworkInput>) {
      VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<64x32x32x16xf16, @DDR> :  swizzling(0)>
    }
    // CHECK:    ELF.CreateLogicalSection @io.NetworkOutput0
    // CHECK-SAME:    secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USEROUTPUT|VPU_SHF_PROC_SHAVE")
    ELF.CreateLogicalSection @io.NetworkOutput0 aligned(1) secType(SHT_NOBITS) secFlags("VPU_SHF_USEROUTPUT|SHF_WRITE|SHF_ALLOC") secLocation(<NetworkOutput>) {
      VPUASM.DeclareBuffer @DeclareBuffer3 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<64x32x32x16xf16, @DDR> :  swizzling(0)>
    }
    VPUASM.DeclareKernelEntry @DeclareKernelEntry_0_0 : "eltwise_min"

    // CHECK:    ELF.CreateSection @buffer.Constant.0.constant
    // CHECK-SAME:    secFlags("SHF_ALLOC|VPU_SHF_PROC_SHAVE")
    ELF.CreateSection @buffer.Constant.0.constant aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
      VPUASM.ConstBuffer @Declare0 !VPUASM.Buffer< "Constant"[0] <0> : memref<64x32x32x16xf16> :  swizzling(0)> = dense<1.000000e+00> : tensor<64x32x32x16xf16>
    }
    // CHECK:   ELF.CreateSection @shave.text
    // CHECK-SAME:    secFlags("SHF_ALLOC|VPU_SHF_PROC_SHAVE")
    ELF.CreateSection @shave.text aligned(1024) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
      VPUASM.DeclareKernelText @DeclareKernelText_0_0 : "eltwise_min"
    }
    // CHECK:   ELF.CreateSection @shave.data
    // CHECK-SAME:    secFlags("SHF_ALLOC|VPU_SHF_PROC_SHAVE")
    ELF.CreateSection @shave.data aligned(1024) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
      VPUASM.DeclareKernelData @DeclareKernelArgs_0_0 : "eltwise_min"
    }
    // CHECK:   ELF.CreateSection @shave.params aligned(1024)
    // CHECK-SAME:    secFlags("SHF_ALLOC|VPU_SHF_PROC_SHAVE")
    ELF.CreateSection @shave.params aligned(1024) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
      VPUASM.KernelParams @KernelParams_0_0 inputs([@io.NetworkInput0::@DeclareBuffer1, @buffer.Constant.0.constant::@Declare0]) outputs([@io.NetworkOutput0::@DeclareBuffer3]) dynamicInputShapes([]) dynamicOutputShapes([]) kernel_type("eltwise_min") < {kernel_params = []}>
      VPUASM.KernelParams @KernelParams_0_1 inputs([]) outputs([]) dynamicInputShapes([]) dynamicOutputShapes([]) kernel_type("cache_op_flush_invalidate") < {kernel_params = [255]}>
    }
    // CHECK:   ELF.CreateSection @program.barrier
    // CHECK-SAME:    secFlags("SHF_ALLOC|SHF_EXECINSTR")
    ELF.CreateSection @program.barrier aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
      VPUASM.ConfigureBarrier @ConfigureBarrier_0_0 idx(!VPURegMapped.Index<0:0:0>) (0) => (-1) counts(1 : 1)
    }
    // CHECK:   ELF.CreateSection @shave.runtime
    // CHECK-SAME:    secFlags("SHF_ALLOC|VPU_SHF_PROC_SHAVE")
    ELF.CreateSection @shave.runtime aligned(1024) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
      VPUASM.ActShaveRt @ActShaveRt kernel("nnActEntry")
    }

    // CHECK:   ELF.CreateSection @note.MappedInferenceVersion
    // CHECK-SAME:    secFlags("SHF_NONE")
    ELF.CreateSection @note.MappedInferenceVersion aligned(4) secType(SHT_NOTE) secFlags("SHF_NONE") secLocation(<DDR>) {
        VPUASM.MappedInferenceVersion @MappedInferenceVersion_0_0(11 _ 4 _ 10)
    }
    // CHECK:   ELF.CreateSection @program.mapped_inference
    // CHECK-SAME:    secFlags("SHF_ALLOC|SHF_EXECINSTR")
    ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
      VPUASM.MappedInference @MappedInference : dmas([]) actKernelRanges([@ActKernelRange_0_0]) actKernelInvocations([@ActKernelInvocation_0_0]) barriers(@program.barrier::@ConfigureBarrier_0_0) actShaveRt(@shave.runtime::@ActShaveRt) dmaCount([[0, 0], [0, 0]]) invariantCount([0, 0]) variantCount([0, 0]) actKernelRangesCount([2, 0]) actKernelInvocationsCount([2, 0]) mediaCount(0) barrierCount(3) mappedInferenceVersion(@note.MappedInferenceVersion::@MappedInferenceVersion_0_0)
    }
    ELF.CreateSection @note.LoaderABIVersion aligned(4) secType(SHT_NOTE) secFlags("SHF_NONE") secLocation(<DDR>) {
      ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    }
    ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
      ELF.Symbol @elfsym.buffer.Constant.0.constant of(@buffer.Constant.0.constant) type(<STT_SECTION>) size(0) value(0)
      ELF.Symbol @elfsym.shave.text of(@shave.text) type(<STT_SECTION>) size(0) value(0)
      ELF.Symbol @elfsym.shave.data of(@shave.data) type(<STT_SECTION>) size(0) value(0)
      ELF.Symbol @elfsym.shave.params of(@shave.params) type(<STT_SECTION>) size(0) value(0)
      ELF.Symbol @elfsym.program.barrier of(@program.barrier) type(<STT_SECTION>) size(0) value(0)
      ELF.Symbol @elfsym.shave.runtime of(@shave.runtime) type(<STT_SECTION>) size(0) value(0)
      ELF.Symbol @elfsym.program.mapped_inference of(@program.mapped_inference) type(<STT_SECTION>) size(0) value(0)
      ELF.Symbol @elfsym.note.LoaderABIVersion of(@note.LoaderABIVersion) type(<STT_SECTION>) size(0) value(0)
    }
    ELF.CreateSymbolTableSection @symtab.io.NetworkInput secFlags(VPU_SHF_USERINPUT) {
      ELF.Symbol @elfsym.io.NetworkInput0 of(@io.NetworkInput0) type(<STT_SECTION>) size(2097152) value(0)
    }
    ELF.CreateSymbolTableSection @symtab.io.NetworkOutput secFlags(VPU_SHF_USEROUTPUT) {
      ELF.Symbol @elfsym.io.NetworkOutput0 of(@io.NetworkOutput0) type(<STT_SECTION>) size(2097152) value(0)
    }
  }
  return
}
}
