//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


// RUN: vpux-opt --init-compiler="platform=%platform%" --convert-VPUASM-to-NPUReg50XX --create-elf-relocations %s | FileCheck %s
// REQUIRES: dev-build && (platform-NPU5010)

module @Model20 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Input" : tensor<1x50x1x1xf16>
  } outputsInfo : {
    DataInfo "Sigmoid_225" : tensor<1x50x1x1xf16>
  }

  func.func @main() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<DDR>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_ActKernelRange_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <ActKernelRange> {elfMemOffsetAttrKey = 51200 : ui64}
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_ActKernelInvocation_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <ActKernelInvocation> {elfMemOffsetAttrKey = 53760 : ui64}
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer5 !VPUASM.Buffer< "CMX_NN"[0] <120> : memref<1x10x2x3xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer6 !VPUASM.Buffer< "CMX_NN"[0] <240> : memref<1x10x2x3xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @buffer.DDR.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
	VPUASM.DeclareBuffer @DeclareBuffer7 !VPUASM.Buffer< "DDR"[0] <0> : memref<0x0x0x0xi32, @DDR> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer8 !VPUASM.Buffer< "DDR"[0] <0> : memref<0x0x0x0xi32, @DDR> :  swizzling(0)>
      }

      VPUASM.DeclareKernelEntry @DeclareKernelEntry_0_0 : "activation_sigmoid"

      ELF.CreateSection @shave.text aligned(1024) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DeclareKernelText @DeclareKernelText_0_0 {elfMemOffsetAttrKey = 0 : ui64} : "activation_sigmoid"
      }
      ELF.CreateSection @shave.data aligned(1024) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DeclareKernelData @DeclareKernelArgs_0_0 {elfMemOffsetAttrKey = 0 : ui64} : "activation_sigmoid"
      }
      ELF.CreateSection @shave.params aligned(1024) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.KernelParams @KernelParams_0_0 inputs([@buffer.CMX_NN.0::@DeclareBuffer5]) outputs([@buffer.CMX_NN.0::@DeclareBuffer6]) dynamicInputShapes([]) dynamicOutputShapes([]) kernel_type("activation_dma_sigmoid") releaseDesc [@task.dma.0.0::@NNDMA_0_0_1, @task.dma.0.1::@NNDMA_0_1_1] {elfMemOffsetAttrKey = 0 : ui64} <
        { inputDimsBinaryVector = [], inputStridesBinaryVector = [], outputDimsBinaryVector = [], outputStridesBinaryVector = [], skipDescIds = [0, 1],
          kernel_params = [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }>
      }
      ELF.CreateSection @task.shave.range.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.ActKernelRange @ActKernelRange_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_ActKernelRange_0_0_0) kernelTaskType(@COMPUTE) calls @shave.text::@DeclareKernelText_0_0 : @DeclareKernelEntry_0_0 {elfMemOffsetAttrKey = 0 : ui64}
      }
      ELF.CreateSection @task.shave.invocation.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.ActKernelInvocation @ActKernelInvocation_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_ActKernelInvocation_0_0_0) -> @program.metadata.cmx::@DeclareTaskBuffer_ActKernelRange_0_0_0(kernel_data : @shave.data::@DeclareKernelArgs_0_0, kernel_params : @shave.params::@KernelParams_0_0) waits([0 : ui16]) updates([1 : ui16]) tile(0) start_after(2) clean_after(1) range_index(0) {elfMemOffsetAttrKey = 0 : ui64}
      }
      ELF.CreateSection @task.dma.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) links(@task.dma.0.0::@NNDMA_0_0_1) input(@buffer.DDR.0::@DeclareBuffer7) outputs([@buffer.DDR.0::@DeclareBuffer8]) waits([]) updates([]) start_after(0) clean_after(0) dma_descriptor(<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>) acceleration_mode(<DISABLE>) is_out_of_order() {skip_dma = #VPUIP.SkipDMAAttr<tile = 0 : i64, list = 0 : i64, logicalTask = 0 : i64, descId = 0 : i64>}
        VPUASM.NNDMA @NNDMA_0_0_1 idx(!VPURegMapped.Index<0:0:1>) input(@buffer.DDR.0::@DeclareBuffer7) outputs([@buffer.DDR.0::@DeclareBuffer8]) waits([]) updates([]) start_after(0) clean_after(0) dma_descriptor(<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>) acceleration_mode(<DISABLE>) is_out_of_order()
      }
      ELF.CreateSection @task.dma.0.1 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.NNDMA @NNDMA_0_1_0 idx(!VPURegMapped.Index<0:1:0>) links(@task.dma.0.1::@NNDMA_0_1_1) input(@buffer.DDR.0::@DeclareBuffer7) outputs([@buffer.DDR.0::@DeclareBuffer8]) waits([]) updates([]) start_after(0) clean_after(0) dma_descriptor(<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>) acceleration_mode(<DISABLE>) is_out_of_order() {skip_dma = #VPUIP.SkipDMAAttr<tile = 0 : i64, list = 1 : i64, logicalTask = 0 : i64, descId = 1 : i64>}
        VPUASM.NNDMA @NNDMA_0_1_1 idx(!VPURegMapped.Index<0:1:1>) input(@buffer.DDR.0::@DeclareBuffer7) outputs([@buffer.DDR.0::@DeclareBuffer8]) waits([]) updates([]) start_after(0) clean_after(0) dma_descriptor(<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>) acceleration_mode(<DISABLE>) is_out_of_order()
      }
      ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
        ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>) size(82944) value(1075854336)
	ELF.Symbol @elfsym.buffer.DDR.0 of(@buffer.DDR.0) type(<STT_SECTION>)
        ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>) size(1474560) value(1075937280)
        ELF.Symbol @elfsym.shave.text of(@shave.text) type(<STT_SECTION>)
        ELF.Symbol @elfsym.shave.data of(@shave.data) type(<STT_SECTION>)
        ELF.Symbol @elfsym.shave.params of(@shave.params) type(<STT_SECTION>)
        ELF.Symbol @elfsym.task.shave.range.0.0 of(@task.shave.range.0.0) type(<STT_SECTION>)
        ELF.Symbol @elfsym.task.shave.invocation.0.0 of(@task.shave.invocation.0.0) type(<STT_SECTION>)
	ELF.Symbol @elfsym.task.dma.0.0 of(@task.dma.0.0) type(<STT_SECTION>)
        ELF.Symbol @elfsym.task.dma.0.1 of(@task.dma.0.1) type(<STT_SECTION>)
      }

    }
    return
}

    // CHECK: ELF.CreateSection @shave.params aligned(1024) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
    // CHECK:    VPUASM.KernelParams
    // CHECK: releaseDesc [@task.dma.0.0::@NNDMA_0_0_1, @task.dma.0.1::@NNDMA_0_1_1] {elfMemOffsetAttrKey = 0 : ui64}
    // CHECK: <{inputDimsBinaryVector = [], inputStridesBinaryVector = []
    // CHECK: kernel_params = [120, 128, 1, 64, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 240, 128, 1, 64, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    // CHECK: outputDimsBinaryVector = [], outputStridesBinaryVector = [], skipDescIds = [0, 1]}>

    // CHECK:      ELF.Reloc
    // CHECK-SAME:   offset({{[0-9]+}})
    // CHECK-SAME:   sourceSym(@symtab::@elfsym.task.dma.0.0)
    // CHECK-SAME:   relocType(<R_VPU_64>)
    // CHECK-SAME:   addend({{[0-9]+}})
    // CHECK-SAME:   (description : "Release desc 0 kernel params reloc")
    // CHECK:      ELF.Reloc
    // CHECK-SAME:   offset({{[0-9]+}})
    // CHECK-SAME:   sourceSym(@symtab::@elfsym.task.dma.0.1)
    // CHECK-SAME:   relocType(<R_VPU_64>)
    // CHECK-SAME:   addend({{[0-9]+}})
    // CHECK-SAME:   (description : "Release desc 1 kernel params reloc")
}
