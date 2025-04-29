//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --convert-VPUMI37XX-to-ELF %s -o stride_dma_vpumi.mlir
// RUN: vpux-translate --vpu-arch=%arch% --export-ELF stride_dma_vpumi.mlir -o stride_dma.elf
// RUN: vpux-translate --vpu-arch=%arch% --import-ELF stride_dma.elf | FileCheck %s
// RUN: rm stride_dma.elf stride_dma_vpumi.mlir
// REQUIRES: arch-NPU37XX
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @mainModule {
  IE.CNNNetwork entryPoint : @dma_src_dst_all_with_stride inputsInfo : {
    DataInfo "input_0" : tensor<1x3x31x53xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x3x31x53xf16>
  }
  func.func private @dma_src_dst_all_with_stride(%arg0: memref<1x3x31x53xf16, {order = #NCHW, strides = [12288, 4096, 64, 1]}, @DDR>, %arg1: memref<1x3x31x53xf16, #NCHW, @DDR>) -> memref<1x3x31x53xf16, #NCHW, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x31x53xf16, {order = #NCHW, strides = [5952, 1984, 64, 1]}, [@CMX_NN, 0]>
    %1 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    %3 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x3x31x53xf16, {order = #NCHW, strides = [5952, 1984, 64, 1]}, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x3x31x53xf16, #NCHW, @DDR>) waits(%1 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %2 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x3x31x53xf16, {order = #NCHW, strides = [12288, 4096, 64, 1]}, @DDR>) outputs(%0 : memref<1x3x31x53xf16, {order = #NCHW, strides = [5952, 1984, 64, 1]}, [@CMX_NN, 0]>) nextDMAIdx(%3 : !VPURegMapped.Index<0:0:1>) updates(%1 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %4 = VPUMI37XX.MappedInference dmas(%2 : !VPURegMapped.Index<0:0:0>) barriers(%1 : !VPURegMapped.Index<0:0:0>) dmaCount([2, 0]) invariantCount(0) variantCount(0) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(1) -> !VPURegMapped.Index<0:0:0>

    return %arg1 : memref<1x3x31x53xf16, #NCHW, @DDR>
  }


//CHECK:      [[BAR0:%.+]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, 4294967295> -> !VPURegMapped.Index<0:0:0>
//CHECK-DAG:  [[BAR_CFG:%.+]] = ELFNPU37XX.CreateSection {{.*}} secName = ".text.BarrierConfigs"
//CHECK-NEXT: ELFNPU37XX.PutOpInSection [[BAR0]] : !VPURegMapped.Index<0:0:0>

//CHECK:     [[CMX_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<106xui8, [@CMX_NN, 0]>

//CHECK:     [[NNDMA_1:%.+]] = VPUMI37XX.NNDMA {
//CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<
//CHECK-SAME:         numPlanes = 0 : i64, len = 9858 : i64
//CHECK-SAME:         srcWidth = 106 : i64, srcStride = 128 : i64, srcPlaneStride = 0 : i64
//CHECK-SAME:         dstWidth = 9858 : i64, dstStride = 9858 : i64, dstPlaneStride = 0 : i64>
//CHECK-SAME:         is_critical, is_out_of_order, port = 0 : si64}
//CHECK:            inputs([[CMX_BUF]] : memref<106xui8, [@CMX_NN, 0]>) outputs(%arg1 : memref<9858xui8>) waits([[BAR0]] : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>

//CHECK:     [[NNDMA_0:%.+]] = VPUMI37XX.NNDMA {
//CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<
//CHECK-SAME:         numPlanes = 2 : i64, len = 3286 : i64
//CHECK-SAME:         srcWidth = 106 : i64, srcStride = 128 : i64, srcPlaneStride = 8192 : i64
//CHECK-SAME:         dstWidth = 106 : i64, dstStride = 128 : i64, dstPlaneStride = 3968 : i64>
//CHECK-SAME:         is_critical, is_out_of_order, port = 0 : si64} inputs(%arg0 : memref<24576xui8>)
//CHECK:           outputs([[CMX_BUF]] : memref<106xui8, [@CMX_NN, 0]>) nextDMAIdx([[NNDMA_1]] : !VPURegMapped.Index<0:0:1>) updates([[BAR0]] : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>


//CHECK:    [[DMA0_TEXT:%.+]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks0"} -> !ELFNPU37XX.Section {
//CHECK-NEXT:      ELFNPU37XX.PutOpInSection [[NNDMA_0]] : !VPURegMapped.Index<0:0:0>
//CHECK-NEXT:      ELFNPU37XX.PutOpInSection [[NNDMA_1]] : !VPURegMapped.Index<0:0:1>

//CHECK:    [[MAPINFER:%.+]] = VPUMI37XX.MappedInference dmas([[NNDMA_0]] : !VPURegMapped.Index<0:0:0>) barriers([[BAR0]] : !VPURegMapped.Index<0:0:0>) dmaCount([2]) invariantCount(0) variantCount(0) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(1) -> !VPURegMapped.Index<0:0:0>

//CHECK:    [[MAPINFER_TEXT:%.+]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.MappedInference"} -> !ELFNPU37XX.Section {
//CHECK-NEXT:      ELFNPU37XX.PutOpInSection [[MAPINFER]] : !VPURegMapped.Index<0:0:0>

//CHECK:    ELFNPU37XX.CreateMetadataSection secFlags("SHF_NONE") {secAddrAlign = 8 : i64, secInfo = 0 : i64, secName = ".metadata"} -> !ELFNPU37XX.Section {
//CHECK-NEXT:      VPUMI37XX.NetworkMetadata -> !VPURegMapped.Index<0:0:0>

//CHECK:    [[DMA1_TEXT:%.+]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks1"} -> !ELFNPU37XX.Section {

//CHECK:    [[INPUT_0:%.+]] = ELFNPU37XX.Symbol %arg0 name("input_0") type(<STT_NOTYPE>) size(24576) {value = 0 : ui64} : memref<24576xui8>
//CHECK:    [[SYM_INPUT:%.+]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELFNPU37XX.Section {
//CHECK-NEXT:      ELFNPU37XX.PutOpInSection [[INPUT_0]] : !ELFNPU37XX.Symbol

//CHECK-DAG:    [[OUTPUT_0:%.+]] = ELFNPU37XX.Symbol %arg1 name("output_0") type(<STT_NOTYPE>) size(9858) {value = 0 : ui64} : memref<9858xui8>
//CHECK-DAG:    [[SYM_OUTPUT:%.+]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELFNPU37XX.Section {
//CHECK-DAG:      ELFNPU37XX.PutOpInSection [[OUTPUT_0]] : !ELFNPU37XX.Symbol

//CHECK-DAG:    [[SYM_DMA_0:%.+]] = ELFNPU37XX.Symbol %6 name("sym_dmaSection0") type(<STT_NOTYPE>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
//CHECK-DAG:    [[SYM_BAR_SEC:%.+]] = ELFNPU37XX.Symbol [[BAR_CFG]] name("sym_barrierSection") type(<STT_NOTYPE>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
//CHECK-DAG:    [[SYM_INFER_ENTRY:%.+]] = ELFNPU37XX.Symbol [[MAPINFER_TEXT]] name("MappedInference_entry") type(<VPU_STT_ENTRY>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
//CHECK-DAG:    %37 = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.tasks") secFlags("SHF_NONE") -> !ELFNPU37XX.Section {
//CHECK-DAG:      ELFNPU37XX.PutOpInSection [[SYM_DMA_0]] : !ELFNPU37XX.Symbol
//CHECK-DAG:      ELFNPU37XX.PutOpInSection [[SYM_BAR_SEC]] : !ELFNPU37XX.Symbol
//CHECK-DAG:      ELFNPU37XX.PutOpInSection [[SYM_INFER_ENTRY]] : !ELFNPU37XX.Symbol

//CHECK:    [[CONST_0:%.+]] = arith.constant 0 : i8
//CHECK:    [[SYM_BASE_ADDR:%.+]] = ELFNPU37XX.Symbol [[CONST_0]] name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") {isBuiltin} : i8
//CHECK:    [[CONST_1:%.+]] = arith.constant 1 : i8
//CHECK:    [[SYM_RTM_IVAR:%.+]] = ELFNPU37XX.Symbol [[CONST_1]] name("VPU_NNRD_SYM_RTM_IVAR") {isBuiltin} : i8
//CHECK:    [[CONST_2:%.+]] = arith.constant 2 : i8
//CHECK:    [[SYM_RTM_ACT:%.+]] = ELFNPU37XX.Symbol [[CONST_2]] name("VPU_NNRD_SYM_RTM_ACT") {isBuiltin} : i8
//CHECK:    [[CONST_3:%.+]] = arith.constant 3 : i8
//CHECK:    [[SYM_RTM_DMA0:%.+]] = ELFNPU37XX.Symbol [[CONST_3]] name("VPU_NNRD_SYM_RTM_DMA0") {isBuiltin} : i8
//CHECK:    [[CONST_4:%.+]] = arith.constant 4 : i8
//CHECK:    [[SYM_RTM_DMA1:%.+]] = ELFNPU37XX.Symbol [[CONST_4]] name("VPU_NNRD_SYM_RTM_DMA1") {isBuiltin} : i8
//CHECK:    [[CONST_5:%.+]] = arith.constant 5 : i8
//CHECK:    [[SYM_FIFO_BASE:%.+]] = ELFNPU37XX.Symbol [[CONST_5]] name("VPU_NNRD_SYM_FIFO_BASE") {isBuiltin} : i8
//CHECK:    [[CONST_6:%.+]] = arith.constant 6 : i8
//CHECK:    [[BAR_START:%.+]] = ELFNPU37XX.Symbol [[CONST_6]] name("VPU_NNRD_SYM_BARRIERS_START") {isBuiltin} : i8
//CHECK:    [[CONST_7:%.+]] = arith.constant 7 : i8
//CHECK:    [[HW_REG:%.+]] = ELFNPU37XX.Symbol [[CONST_7]] name("VPU_NNRD_SYM_HW_REGISTER") {isBuiltin} : i8
//CHECK:    [[SYMSEC_RT:%.+]] = ELFNPU37XX.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags("SHF_NONE") {isBuiltin} -> !ELFNPU37XX.Section {
//CHECK-NEXT:      ELFNPU37XX.PutOpInSection [[SYM_BASE_ADDR]] : !ELFNPU37XX.Symbol
//CHECK-NEXT:      ELFNPU37XX.PutOpInSection [[SYM_RTM_IVAR]] : !ELFNPU37XX.Symbol
//CHECK-NEXT:      ELFNPU37XX.PutOpInSection [[SYM_RTM_ACT]] : !ELFNPU37XX.Symbol
//CHECK-NEXT:      ELFNPU37XX.PutOpInSection [[SYM_RTM_DMA0]] : !ELFNPU37XX.Symbol
//CHECK-NEXT:      ELFNPU37XX.PutOpInSection [[SYM_RTM_DMA1]] : !ELFNPU37XX.Symbol
//CHECK-NEXT:      ELFNPU37XX.PutOpInSection [[SYM_FIFO_BASE]] : !ELFNPU37XX.Symbol
//CHECK-NEXT:      ELFNPU37XX.PutOpInSection [[BAR_START]] : !ELFNPU37XX.Symbol
//CHECK-NEXT:      ELFNPU37XX.PutOpInSection [[HW_REG]] : !ELFNPU37XX.Symbol

//CHECK:    ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetInput0") sourceSymbolTableSection([[SYM_INPUT]]) targetSection(%6) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELFNPU37XX.Section {
//CHECK:      ELFNPU37XX.Reloc offset(16) <R_VPU_64> [[INPUT_0]] 0

//CHECK:    ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetOutput0") sourceSymbolTableSection([[SYM_OUTPUT]]) targetSection(%6) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELFNPU37XX.Section {
//CHECK:      ELFNPU37XX.Reloc offset(152) <R_VPU_64> [[OUTPUT_0]] 0

//CHECK:    ELFNPU37XX.CreateRelocationSection secName(".rlt.text.dmaTasks0") sourceSymbolTableSection([[SYMSEC_RT]]) targetSection(%6) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
//CHECK:      ELFNPU37XX.Reloc offset(24) <R_VPU_64> [[SYM_BASE_ADDR]] 0
//CHECK:      ELFNPU37XX.Reloc offset(0) <R_VPU_32_RTM> [[SYM_RTM_DMA0]] 128
//CHECK:      ELFNPU37XX.Reloc offset(64) <R_VPU_64_LSHIFT>
//CHECK:      ELFNPU37XX.Reloc offset(72) <R_VPU_64_LSHIFT>
//CHECK:      ELFNPU37XX.Reloc offset(144) <R_VPU_64> [[SYM_BASE_ADDR]] 0
//CHECK:      ELFNPU37XX.Reloc offset(192) <R_VPU_64_LSHIFT>
//CHECK:      ELFNPU37XX.Reloc offset(200) <R_VPU_64_LSHIFT>


//CHECK:    ELFNPU37XX.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%37) targetSection(%8) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
//CHECK:      ELFNPU37XX.Reloc offset(72) <R_VPU_64> %27 0
//CHECK:      ELFNPU37XX.Reloc offset(112) <R_VPU_64> [[SYM_DMA_0]] 0
//CHECK:      ELFNPU37XX.Reloc offset(312) <R_VPU_64> [[SYM_BAR_SEC]] 0

//CHECK: ELFNPU37XX.CreateRelocationSection secName(".rlt.text.BarrierConfigs") sourceSymbolTableSection(%46) targetSection(%1) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
//CHECK: ELFNPU37XX.Reloc offset(8) <R_VPU_16_SUM>


// single tile concurent execution
//CHECK: ELFNPU37XX.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%46) targetSection(%8) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section
//CHECK:      ELFNPU37XX.Reloc offset(72) <R_VPU_64_MULT_SUB>
//CHECK:      ELFNPU37XX.Reloc offset(80) <R_VPU_64_MULT_SUB>
//CHECK:      ELFNPU37XX.Reloc offset(112) <R_VPU_64_MULT>
//CHECK:      ELFNPU37XX.Reloc offset(120) <R_VPU_64_MULT>


//CHECK-DAG:    return %arg1 : memref<9858xui8>
}
