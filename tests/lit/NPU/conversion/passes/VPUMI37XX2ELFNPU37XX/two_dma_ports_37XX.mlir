//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUMI37XX-to-ELF %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {
  net.NetworkInfo entryPoint : @race_condition_dma_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x16x16xf16>
    DataInfo "output_1" : tensor<1x16x16x16xf16>
  }

  // CHECK-LABEL: func.func nested @race_condition_dma_f16_f16
  // CHECK-SAME:    [[ARG_0:%[^:]+]]: memref<1x16x16x16xf16, {order = #NHWC}, @DDR>
  // CHECK-SAME:    [[ARG_1:%[^:]+]]: memref<1x16x16x16xf16, {order = #NHWC}, @DDR>
  // CHECK-SAME:    [[ARG_2:%[^:]+]]: memref<1x16x16x16xf16, {order = #NHWC}, @DDR>
  func.func nested @race_condition_dma_f16_f16(%arg0: memref<1x16x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x16x16x16xf16, {order = #NHWC}, @DDR>, %arg2: memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) -> (memref<1x16x16x16xf16, {order = #NHWC}, @DDR>, memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>
    %2 = VPUMI37XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    %5 = VPUMI37XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>
    %9 = VPUMI37XX.NNDMA <{port = 1 : i64}> inputs(%1 : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>) outputs(%arg2 : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) waits(%5 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:2>
    %8 = VPUMI37XX.NNDMA <{port = 0 : i64}> inputs(%0 : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) waits(%5 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %7 = VPUMI37XX.NNDMA <{port = 1 : i64}> inputs(%arg0 : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) outputs(%1 : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>) nextDMAIdx(%9 : !VPURegMapped.Index<0:1:2>) waits(%2 : !VPURegMapped.Index<0:0:0>) updates(%5 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:1>
    %6 = VPUMI37XX.NNDMA <{port = 0 : i64}> inputs(%arg0 : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) outputs(%0 : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) nextDMAIdx(%8 : !VPURegMapped.Index<0:0:2>) waits(%2 : !VPURegMapped.Index<0:0:0>) updates(%5 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %4 = VPUMI37XX.NNDMA <{port = 1 : i64}> inputs(%arg0 : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) outputs(%1 : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>) nextDMAIdx(%7 : !VPURegMapped.Index<0:1:1>) updates(%2 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %3 = VPUMI37XX.NNDMA <{port = 0 : i64}> inputs(%arg0 : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) outputs(%0 : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) nextDMAIdx(%6 : !VPURegMapped.Index<0:0:1>) updates(%2 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %10 = VPUMI37XX.MappedInference dmas(%3, %4 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>) barriers(%2 : !VPURegMapped.Index<0:0:0>) dmaCount([3, 3]) invariantCount(0) variantCount(0) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(2) -> !VPURegMapped.Index<0:0:0>

    // CHECK: [[BAR_0:%.+]] = VPUMI37XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    // CHECK: [[BAR_1:%.+]] = VPUMI37XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>
    // CHECK: [[DMA_P1_2:%.+]] = VPUMI37XX.NNDMA {{.+}}-> !VPURegMapped.Index<0:1:2>
    // CHECK: [[DMA_P0_2:%.+]] = VPUMI37XX.NNDMA {{.+}}-> !VPURegMapped.Index<0:0:2>
    // CHECK: [[DMA_P1_1:%.+]] = VPUMI37XX.NNDMA {{.+}}-> !VPURegMapped.Index<0:1:1>
    // CHECK: [[DMA_P0_1:%.+]] = VPUMI37XX.NNDMA {{.+}}-> !VPURegMapped.Index<0:0:1>
    // CHECK: [[DMA_P1_0:%.+]] = VPUMI37XX.NNDMA {{.+}}-> !VPURegMapped.Index<0:1:0>
    // CHECK: [[DMA_P0_0:%.+]] = VPUMI37XX.NNDMA {{.+}}-> !VPURegMapped.Index<0:0:0>
    // CHECK: [[MI:%.+]] = VPUMI37XX.MappedInference {{.+}}-> !VPURegMapped.Index<0:0:0>

    // CHECK: [[dmaSec0:%.+]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks0"} -> !ELFNPU37XX.Section {
      // CHECK: ELFNPU37XX.PutOpInSection [[DMA_P0_0]] : !VPURegMapped.Index<0:0:0>
      // CHECK: ELFNPU37XX.PutOpInSection [[DMA_P0_1]] : !VPURegMapped.Index<0:0:1>
      // CHECK: ELFNPU37XX.PutOpInSection [[DMA_P0_2]] : !VPURegMapped.Index<0:0:2>

    // CHECK: [[dmaSec1:%.+]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks1"} -> !ELFNPU37XX.Section {
      // CHECK: ELFNPU37XX.PutOpInSection [[DMA_P1_0]] : !VPURegMapped.Index<0:1:0>
      // CHECK: ELFNPU37XX.PutOpInSection [[DMA_P1_1]] : !VPURegMapped.Index<0:1:1>
      // CHECK: ELFNPU37XX.PutOpInSection [[DMA_P1_2]] : !VPURegMapped.Index<0:1:2>

    // CHECK: [[barSec:%.+]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.BarrierConfigs"} -> !ELFNPU37XX.Section {
      // CHECK: ELFNPU37XX.PutOpInSection [[BAR_0]] : !VPURegMapped.Index<0:0:0>
      // CHECK: ELFNPU37XX.PutOpInSection [[BAR_1]] : !VPURegMapped.Index<0:0:1>

    // CHECK: [[miSec:%.+]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.MappedInference"} -> !ELFNPU37XX.Section {
      // CHECK: ELFNPU37XX.PutOpInSection [[MI]] : !VPURegMapped.Index<0:0:0>

    // CHECK: [[metaSec:%.+]] = ELFNPU37XX.CreateMetadataSection secFlags("SHF_NONE") {secAddrAlign = 8 : i64, secInfo = 0 : i64, secName = ".metadata"} -> !ELFNPU37XX.Section {
      // CHECK: [[netMeta:%.+]] = VPUMI37XX.NetworkMetadata -> !VPURegMapped.Index<0:0:0>

    // CHECK: [[symDmaSec0:%.+]] = ELFNPU37XX.Symbol [[dmaSec0]] name("sym_dmaSection0") : !ELFNPU37XX.Section
    // CHECK: [[symDmaSec1:%.+]] = ELFNPU37XX.Symbol [[dmaSec1]] name("sym_dmaSection1") : !ELFNPU37XX.Section
    // CHECK: [[symBarSec:%.+]] = ELFNPU37XX.Symbol [[barSec]] name("sym_barrierSection") : !ELFNPU37XX.Section

    // CHECK: [[symIn0:%.+]] = ELFNPU37XX.Symbol [[ARG_0]] name("input_0") size(8192) : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>
    // CHECK: [[symOut0:%.+]] = ELFNPU37XX.Symbol [[ARG_1]] name("output_0") size(8192) : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>
    // CHECK: [[symOut1:%.+]] = ELFNPU37XX.Symbol [[ARG_2]] name("output_1") size(8192) : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>

    // CHECK: [[symTabSecIn:%.+]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELFNPU37XX.Section {
      // CHECK: ELFNPU37XX.PutOpInSection [[symIn0]] : !ELFNPU37XX.Symbol

    // CHECK: [[symTabSecOut:%.+]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELFNPU37XX.Section {
      // CHECK: ELFNPU37XX.PutOpInSection [[symOut0]] : !ELFNPU37XX.Symbol
      // CHECK: ELFNPU37XX.PutOpInSection [[symOut1]] : !ELFNPU37XX.Symbol

    // CHECK: [[c0:%.+]] = arith.constant 0 : i8
    // CHECK: [[SYM_NNCMX_SLICE_BASE:%.+]] = ELFNPU37XX.Symbol [[c0]] name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") <{isBuiltin}> : i8
    // CHECK: [[c1:%.+]] = arith.constant 1 : i8
    // CHECK: [[SYM_RTM_IVAR:%.+]] = ELFNPU37XX.Symbol [[c1]] name("VPU_NNRD_SYM_RTM_IVAR") <{isBuiltin}> : i8
    // CHECK: [[c2:%.+]] = arith.constant 2 : i8
    // CHECK: [[SYM_RTM_ACT:%.+]] = ELFNPU37XX.Symbol [[c2]] name("VPU_NNRD_SYM_RTM_ACT") <{isBuiltin}> : i8
    // CHECK: [[c3:%.+]] = arith.constant 3 : i8
    // CHECK: [[SYM_RTM_DMA0:%.+]] = ELFNPU37XX.Symbol [[c3]] name("VPU_NNRD_SYM_RTM_DMA0") <{isBuiltin}> : i8
    // CHECK: [[c4:%.+]] = arith.constant 4 : i8
    // CHECK: [[SYM_RTM_DMA1:%.+]] = ELFNPU37XX.Symbol [[c4]] name("VPU_NNRD_SYM_RTM_DMA1") <{isBuiltin}> : i8
    // CHECK: [[c5:%.+]] = arith.constant 5 : i8
    // CHECK: [[SYM_FIFO_BASE:%.+]] = ELFNPU37XX.Symbol [[c5]] name("VPU_NNRD_SYM_FIFO_BASE") <{isBuiltin}> : i8
    // CHECK: [[c6:%.+]] = arith.constant 6 : i8
    // CHECK: [[SYM_BARRIERS_START:%.+]] = ELFNPU37XX.Symbol [[c6]] name("VPU_NNRD_SYM_BARRIERS_START") <{isBuiltin}> : i8

    // CHECK: [[RT_SYMTAB:%.+]] = ELFNPU37XX.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags("SHF_NONE") <{isBuiltin}> -> !ELFNPU37XX.Section {
      // CHECK: ELFNPU37XX.PutOpInSection [[SYM_NNCMX_SLICE_BASE]] : !ELFNPU37XX.Symbol
      // CHECK: ELFNPU37XX.PutOpInSection [[SYM_RTM_IVAR]] : !ELFNPU37XX.Symbol
      // CHECK: ELFNPU37XX.PutOpInSection [[SYM_RTM_ACT]] : !ELFNPU37XX.Symbol
      // CHECK: ELFNPU37XX.PutOpInSection [[SYM_RTM_DMA0]] : !ELFNPU37XX.Symbol
      // CHECK: ELFNPU37XX.PutOpInSection [[SYM_RTM_DMA1]] : !ELFNPU37XX.Symbol
      // CHECK: ELFNPU37XX.PutOpInSection [[SYM_FIFO_BASE]] : !ELFNPU37XX.Symbol
      // CHECK: ELFNPU37XX.PutOpInSection [[SYM_BARRIERS_START]] : !ELFNPU37XX.Symbol

    // CHECK: [[symTabTasks:%.+]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.tasks") secFlags("SHF_NONE") -> !ELFNPU37XX.Section {
      // CHECK: ELFNPU37XX.PutOpInSection [[symDmaSec0]] : !ELFNPU37XX.Symbol
      // CHECK: ELFNPU37XX.PutOpInSection [[symDmaSec1]] : !ELFNPU37XX.Symbol
      // CHECK: ELFNPU37XX.PutOpInSection [[symBarSec]] : !ELFNPU37XX.Symbol
      // CHECK: [[miSym:%.+]] = ELFNPU37XX.Symbol [[MI]] name("MappedInference_entry") type(<VPU_STT_ENTRY>) : !VPURegMapped.Index<0:0:0>

    // CHECK: [[rltDmaNetIn0:%.+]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetInput0") sourceSymbolTableSection([[symTabSecIn]]) targetSection([[dmaSec0]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELFNPU37XX.Section {
      // CHECK: ELFNPU37XX.Reloc baseOp([[DMA_P0_0]] : !VPURegMapped.Index<0:0:0>) offset(16) <R_VPU_64> [[symIn0]] 0
      // CHECK: ELFNPU37XX.Reloc baseOp([[DMA_P0_1]] : !VPURegMapped.Index<0:0:1>) offset(16) <R_VPU_64> [[symIn0]] 0

    // CHECK: [[rltDmaNetOut0:%.+]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetOutput0") sourceSymbolTableSection([[symTabSecOut]]) targetSection([[dmaSec0]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELFNPU37XX.Section {
      // CHECK: ELFNPU37XX.Reloc baseOp([[DMA_P0_2]] : !VPURegMapped.Index<0:0:2>) offset(24) <R_VPU_64> [[symOut0]] 0

    // CHECK: [[rltDmaTasks0:%.+]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.dmaTasks0") sourceSymbolTableSection([[RT_SYMTAB]]) targetSection([[dmaSec0]]) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
      // CHECK: ELFNPU37XX.Reloc baseOp([[DMA_P0_0]] : !VPURegMapped.Index<0:0:0>) offset(24) <R_VPU_64> [[SYM_NNCMX_SLICE_BASE]] 0
      // CHECK: ELFNPU37XX.Reloc baseOp([[DMA_P0_0]] : !VPURegMapped.Index<0:0:0>) offset(0) <R_VPU_32_RTM> [[SYM_RTM_DMA0]] 128
      // CHECK: ELFNPU37XX.Reloc baseOp([[DMA_P0_1]] : !VPURegMapped.Index<0:0:1>) offset(24) <R_VPU_64> [[SYM_NNCMX_SLICE_BASE]] 0
      // CHECK: ELFNPU37XX.Reloc baseOp([[DMA_P0_1]] : !VPURegMapped.Index<0:0:1>) offset(0) <R_VPU_32_RTM> [[SYM_RTM_DMA0]] 128
      // CHECK: ELFNPU37XX.Reloc baseOp([[DMA_P0_2]] : !VPURegMapped.Index<0:0:2>) offset(16) <R_VPU_64> [[SYM_NNCMX_SLICE_BASE]] 0

    // CHECK: [[rltDmaNetIn1:%.+]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetInput1") sourceSymbolTableSection([[symTabSecIn]]) targetSection([[dmaSec1]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELFNPU37XX.Section {
      // CHECK: ELFNPU37XX.Reloc baseOp([[DMA_P1_0]] : !VPURegMapped.Index<0:1:0>) offset(16) <R_VPU_64> [[symIn0]] 0
      // CHECK: ELFNPU37XX.Reloc baseOp([[DMA_P1_1]] : !VPURegMapped.Index<0:1:1>) offset(16) <R_VPU_64> [[symIn0]] 0

    // CHECK: [[rltDmaNetOut1:%.+]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetOutput1") sourceSymbolTableSection([[symTabSecOut]]) targetSection([[dmaSec1]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELFNPU37XX.Section {
      // CHECK: ELFNPU37XX.Reloc baseOp([[DMA_P1_2]] : !VPURegMapped.Index<0:1:2>) offset(24) <R_VPU_64> [[symOut1]] 0

    // CHECK: [[rltDmaTasks1:%.+]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.dmaTasks1") sourceSymbolTableSection([[RT_SYMTAB]]) targetSection([[dmaSec1]]) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
      // CHECK: ELFNPU37XX.Reloc baseOp([[DMA_P1_0]] : !VPURegMapped.Index<0:1:0>) offset(24) <R_VPU_64> [[SYM_NNCMX_SLICE_BASE]] 2097152
      // CHECK: ELFNPU37XX.Reloc baseOp([[DMA_P1_0]] : !VPURegMapped.Index<0:1:0>) offset(0) <R_VPU_32_RTM> [[SYM_RTM_DMA1]] 128
      // CHECK: ELFNPU37XX.Reloc baseOp([[DMA_P1_1]] : !VPURegMapped.Index<0:1:1>) offset(24) <R_VPU_64> [[SYM_NNCMX_SLICE_BASE]] 2097152
      // CHECK: ELFNPU37XX.Reloc baseOp([[DMA_P1_1]] : !VPURegMapped.Index<0:1:1>) offset(0) <R_VPU_32_RTM> [[SYM_RTM_DMA1]] 128
      // CHECK: ELFNPU37XX.Reloc baseOp([[DMA_P1_2]] : !VPURegMapped.Index<0:1:2>) offset(16) <R_VPU_64> [[SYM_NNCMX_SLICE_BASE]] 2097152

    // CHECK: [[rltMi:%.+]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection([[symTabTasks]]) targetSection([[miSec]]) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
      // CHECK: ELFNPU37XX.Reloc baseOp([[MI]] : !VPURegMapped.Index<0:0:0>) offset(72) <R_VPU_64> [[symDmaSec0]] 0
      // CHECK: ELFNPU37XX.Reloc baseOp([[MI]] : !VPURegMapped.Index<0:0:0>) offset(112) <R_VPU_64> [[symDmaSec1]] 0
      // CHECK: ELFNPU37XX.Reloc baseOp([[MI]] : !VPURegMapped.Index<0:0:0>) offset(312) <R_VPU_64> [[symBarSec]] 0

    return %arg1, %arg2 : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>, memref<1x16x16x16xf16, {order = #NHWC}, @DDR>
  }
}
