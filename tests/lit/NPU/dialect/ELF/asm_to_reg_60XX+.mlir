//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --convert-VPUASM-to-NPUReg60XX --handle-alignment-requirements %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {
  config.ExecutorResource 1 of @DMA_NN
  config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @read_after_write_act_dma_f16_f16 inputsInfo : {
    DataInfo "input" : tensor<1x10x2x3xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x10x2x3xf16>
  }
  func.func @read_after_write_act_dma_f16_f16() {
    ELF.Main {
      VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x10x2x3xf16, {order = #NHWC}, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x10x2x3xf16, {order = #NHWC}, @DDR> :  swizzling(0)>
      ELF.CreateLogicalSection @program.DMA.cmx.0.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") secLocation(<DDR>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DMA>
      }

      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") secLocation(<DDR>) {
        VPUASM.DeclareBuffer @DeclareBuffer2 !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x10x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dma.0.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") secLocation(<DDR>) {
        VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.DMA.cmx.0.0::@DeclareTaskBuffer_DMA_0_0_0) input(@DeclareBuffer0) outputs([@buffer.CMX_NN.0::@DeclareBuffer2]) waits([]) updates([0 : ui16]) start_after(1) clean_after(0) dma_descriptor(<numPlanes = 0 : i4, len = -8 : i4, srcWidth = -8 : i4, srcStride = -8 : i4, srcPlaneStride = 0 : i4, dstWidth = -8 : i4, dstStride = -8 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>)
      }

      ELF.CreateSection @note.MappedInferenceVersion aligned(4) secType(SHT_NOTE) secFlags("SHF_NONE") secLocation(<DDR>) {
        VPUASM.MappedInferenceVersion @MappedInferenceVersion_0_0(11 _ 4 _ 10)
      }
      ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") secLocation(<DDR>) {
        VPUASM.MappedInference @MappedInference : dmas([[@task.dma.0.0::@NNDMA_0_0_0, @task.dma.0.1::@NNDMA_0_1_0]]) actKernelRanges([@task.shave.range.0.0::@ActKernelRange0]) actKernelInvocations([@task.shave.invocation.0.0::@ActKernelInvocation0]) barriers(@program.barrier::@ConfigureBarrier0) dmaCount([[1, 7], [0, 0]]) invariantCount([0, 0]) variantCount([0, 0]) actKernelRangesCount([6, 0]) actKernelInvocationsCount([6, 0]) barrierCount(13) mappedInferenceVersion(@note.MappedInferenceVersion::@MappedInferenceVersion_0_0)
      }
      ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
        ELF.Symbol @elfsym.program.DMA.cmx.0.0 of(@program.DMA.cmx.0.0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.program.barrier of(@program.barrier) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.task.dma.0.0 of(@task.dma.0.0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.note.MappedInferenceVersion of(@note.MappedInferenceVersion) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.program.mapped_inference of(@program.mapped_inference) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @entry of(@program.mapped_inference::@MappedInference) type(<VPU_STT_ENTRY>) size(0) value(0)
      }
      ELF.CreateMetadataSection @MetadataSection aligned(8) secFlags("SHF_NONE")  {
        VPUASM.NetworkMetadata @NetworkMetadata
      }
    }
    return
  }
  // MappedInferenceOp erased from IR
  // CHECK: ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") secLocation(<DDR>) {
  // CHECK-NOT: VPUASM.MappedInference
  // CHECK: }
}
