//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --convert-VPUMI37XX-to-ELF %s -o %t.mlir
// RUN: vpux-translate --vpu-arch=%arch% --export-ELF %t.mlir -o %t.elf
// RUN: vpux-translate --vpu-arch=%arch% --import-ELF %t.elf | FileCheck %s
// RUN: rm %t.elf %t.mlir
// REQUIRES: arch-NPU37XX
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {
  IE.CNNNetwork entryPoint : @race_condition_dma_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x16x16xf16>
    DataInfo "output_1" : tensor<1x16x16x16xf16>
  }
  func.func private @race_condition_dma_f16_f16(%arg0: memref<1x16x16x16xf16, #NHWC, @DDR>, %arg1: memref<1x16x16x16xf16, #NHWC, @DDR>, %arg2: memref<1x16x16x16xf16, #NHWC, @DDR>) -> (memref<1x16x16x16xf16, #NHWC, @DDR>, memref<1x16x16x16xf16, #NHWC, @DDR>) {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>
    %2 = VPUMI37XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    %5 = VPUMI37XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>
    %9 = VPUMI37XX.NNDMA {port = 1 : i64} inputs(%1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) outputs(%arg2 : memref<1x16x16x16xf16, #NHWC, @DDR>) waits(%5 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:2>
    %8 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x16x16x16xf16, #NHWC, @DDR>) waits(%5 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %7 = VPUMI37XX.NNDMA {port = 1 : i64} inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) nextDMAIdx(%9 : !VPURegMapped.Index<0:1:2>) waits(%2 : !VPURegMapped.Index<0:0:0>) updates(%5 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:1>
    %6 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) nextDMAIdx(%8 : !VPURegMapped.Index<0:0:2>) waits(%2 : !VPURegMapped.Index<0:0:0>) updates(%5 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %4 = VPUMI37XX.NNDMA {port = 1 : i64} inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) nextDMAIdx(%7 : !VPURegMapped.Index<0:1:1>) updates(%2 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %3 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) nextDMAIdx(%6 : !VPURegMapped.Index<0:0:1>) updates(%2 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %10 = VPUMI37XX.MappedInference dmas(%3, %4 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>) barriers(%2 : !VPURegMapped.Index<0:0:0>) dmaCount([3, 3]) invariantCount(0) variantCount(0) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(2) -> !VPURegMapped.Index<0:0:0>

    return %arg1, %arg2 : memref<1x16x16x16xf16, #NHWC, @DDR>, memref<1x16x16x16xf16, #NHWC, @DDR>
  }
}

//CHECK: %[[VAL0:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<0, 4294967295> -> !VPURegMapped.Index<0:0:0>
//CHECK: %[[VAL1:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<1, 4294967295> -> !VPURegMapped.Index<0:0:1>
//CHECK-DAG: %[[VAL2:.*]] = ELFNPU37XX.CreateSection {{.*}} secName = ".text.BarrierConfigs"
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL0]] : !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL1]] : !VPURegMapped.Index<0:0:1>

//CHECK: %[[VAL3:.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<8192xui8, [@CMX_NN, 0]>
//CHECK: %[[VAL6:.*]] = VPUMI37XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 8192 : i64, srcWidth = 8192 : i64, srcStride = 8192 : i64, srcPlaneStride = 0 : i64, dstWidth = 8192 : i64, dstStride = 8192 : i64, dstPlaneStride = 0 : i64>, is_critical, is_out_of_order, port = 0 : si64} inputs(%3 : memref<8192xui8, [@CMX_NN, 0]>) outputs(%[[VALOUTPUT0:.*]] : {{.*}}) waits(%1 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
//CHECK: %[[VAL5:.*]] = VPUMI37XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 8192 : i64, srcWidth = 8192 : i64, srcStride = 8192 : i64, srcPlaneStride = 0 : i64, dstWidth = 8192 : i64, dstStride = 8192 : i64, dstPlaneStride = 0 : i64>, is_critical, is_out_of_order, port = 0 : si64} inputs(%[[VALINPUT0:.*]] : {{.*}}) outputs(%[[VAL3]] : memref<8192xui8, [@CMX_NN, 0]>) nextDMAIdx(%[[VAL6]] : !VPURegMapped.Index<0:0:2>) waits(%[[VAL0]] : !VPURegMapped.Index<0:0:0>) updates(%[[VAL1]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
//CHECK: %[[VAL4:.*]] = VPUMI37XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 8192 : i64, srcWidth = 8192 : i64, srcStride = 8192 : i64, srcPlaneStride = 0 : i64, dstWidth = 8192 : i64, dstStride = 8192 : i64, dstPlaneStride = 0 : i64>, is_critical, is_out_of_order, port = 0 : si64} inputs(%[[VALINPUT0]] : {{.*}}) outputs(%[[VAL3]] : memref<8192xui8, [@CMX_NN, 0]>) nextDMAIdx(%[[VAL5]] : !VPURegMapped.Index<0:0:1>) updates(%[[VAL0]] : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

//CHECK: %[[VAL7:.*]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks0"} -> !ELFNPU37XX.Section {
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL4]] : !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL5]] : !VPURegMapped.Index<0:0:1>
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL6]] : !VPURegMapped.Index<0:0:2>

//CHECK: %[[VAL8:.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <2097152> -> memref<8192xui8, [@CMX_NN, 0]>
//CHECK: %[[VAL11:.*]] = VPUMI37XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 8192 : i64, srcWidth = 8192 : i64, srcStride = 8192 : i64, srcPlaneStride = 0 : i64, dstWidth = 8192 : i64, dstStride = 8192 : i64, dstPlaneStride = 0 : i64>, is_critical, is_out_of_order, port = 0 : si64} inputs(%[[VAL8]] : memref<8192xui8, [@CMX_NN, 0]>) outputs(%[[VALOUTPUT1:.*]] : {{.*}}) waits(%[[VAL1]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
//CHECK: %[[VAL10:.*]] = VPUMI37XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 8192 : i64, srcWidth = 8192 : i64, srcStride = 8192 : i64, srcPlaneStride = 0 : i64, dstWidth = 8192 : i64, dstStride = 8192 : i64, dstPlaneStride = 0 : i64>, is_critical, is_out_of_order, port = 0 : si64} inputs(%[[VALOUTPUT0]] : {{.*}}) outputs(%[[VAL8]] : memref<8192xui8, [@CMX_NN, 0]>) nextDMAIdx(%[[VAL11]] : !VPURegMapped.Index<0:0:2>) waits(%[[VAL0]] : !VPURegMapped.Index<0:0:0>) updates(%[[VAL1]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
//CHECK: %[[VAL9:.*]] = VPUMI37XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 8192 : i64, srcWidth = 8192 : i64, srcStride = 8192 : i64, srcPlaneStride = 0 : i64, dstWidth = 8192 : i64, dstStride = 8192 : i64, dstPlaneStride = 0 : i64>, is_critical, is_out_of_order, port = 0 : si64} inputs(%[[VALOUTPUT0]] : {{.*}}) outputs(%[[VAL8]] : memref<8192xui8, [@CMX_NN, 0]>) nextDMAIdx(%[[VAL10]] : !VPURegMapped.Index<0:0:1>) updates(%[[VAL0]] : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

//CHECK: %[[VAL12:.*]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks1"} -> !ELFNPU37XX.Section {
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL9]] : !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL10]] : !VPURegMapped.Index<0:0:1>
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL11]] : !VPURegMapped.Index<0:0:2>

//CHECK: %[[VAL13:.*]] = VPUMI37XX.MappedInference dmas(%[[VAL4]], %[[VAL9]] : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>) barriers(%[[VAL0]] : !VPURegMapped.Index<0:0:0>) dmaCount([3, 3]) invariantCount(0) variantCount(0) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(2) -> !VPURegMapped.Index<0:0:0>
//CHECK: %[[VAL14:.*]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.MappedInference"} -> !ELFNPU37XX.Section {
//CHECK: ELFNPU37XX.PutOpInSection  %[[VAL13]] : !VPURegMapped.Index<0:0:0>

//CHECK-DAG: %[[symIn0:.*]] = ELFNPU37XX.Symbol %[[VALINPUT0]] name("input_0") {{.*}}
//CHECK-DAG: %[[symTabSecIn:.*]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELFNPU37XX.Section {
//CHECK-DAG: ELFNPU37XX.PutOpInSection %[[symIn0]] : !ELFNPU37XX.Symbol

//CHECK-DAG: %[[symOut0:.*]] = ELFNPU37XX.Symbol %[[VALOUTPUT0]] name("output_0") {{.*}}
//CHECK-DAG: %[[symOut1:.*]] = ELFNPU37XX.Symbol %[[VALOUTPUT1]] name("output_1") {{.*}}
//CHECK-DAG: %[[symTabSecOut:.*]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELFNPU37XX.Section {
//CHECK-DAG: ELFNPU37XX.PutOpInSection %[[symOut0]] : !ELFNPU37XX.Symbol
//CHECK-DAG: ELFNPU37XX.PutOpInSection %[[symOut1]] : !ELFNPU37XX.Symbol

//CHECK-DAG: %[[symDmaSec0:.*]] = ELFNPU37XX.Symbol %[[VAL7]] name("sym_dmaSection0") {{.*}} : !ELFNPU37XX.Section
//CHECK-DAG: %[[symDmaSec1:.*]] = ELFNPU37XX.Symbol %[[VAL12]] name("sym_dmaSection1") {{.*}} : !ELFNPU37XX.Section
//CHECK-DAG: %[[symBarSec:.*]] = ELFNPU37XX.Symbol %[[VAL2]] name("sym_barrierSection") {{.*}} : !ELFNPU37XX.Section

//CHECK-DAG: %[[miSym:.*]] = ELFNPU37XX.Symbol %[[VAL14]] name("MappedInference_entry") type(<VPU_STT_ENTRY>) {{.*}} : !ELFNPU37XX.Section
//CHECK-DAG: %[[symTabTasks:.*]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.tasks") secFlags("SHF_NONE") -> !ELFNPU37XX.Section {
//CHECK-DAG: ELFNPU37XX.PutOpInSection %[[symDmaSec0]] : !ELFNPU37XX.Symbol
//CHECK-DAG: ELFNPU37XX.PutOpInSection %[[symDmaSec1]] : !ELFNPU37XX.Symbol
//CHECK-DAG: ELFNPU37XX.PutOpInSection %[[symBarSec]] : !ELFNPU37XX.Symbol

//CHECK: %[[VALC0:.*]] = arith.constant 0 : i8
//CHECK: %[[VAL60:.*]] = ELFNPU37XX.Symbol %[[VALC0]] name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") {isBuiltin} : i8
//CHECK: %[[VALC1:.*]] = arith.constant 1 : i8
//CHECK: %[[VAL61:.*]] = ELFNPU37XX.Symbol %[[VALC1]] name("VPU_NNRD_SYM_RTM_IVAR") {isBuiltin} : i8
//CHECK: %[[VALC2:.*]] = arith.constant 2 : i8
//CHECK: %[[VAL62:.*]] = ELFNPU37XX.Symbol %[[VALC2]] name("VPU_NNRD_SYM_RTM_ACT") {isBuiltin} : i8
//CHECK: %[[VALC3:.*]] = arith.constant 3 : i8
//CHECK: %[[VAL63:.*]] = ELFNPU37XX.Symbol %[[VALC3]] name("VPU_NNRD_SYM_RTM_DMA0") {isBuiltin} : i8
//CHECK: %[[VALC4:.*]] = arith.constant 4 : i8
//CHECK: %[[VAL64:.*]] = ELFNPU37XX.Symbol %[[VALC4]] name("VPU_NNRD_SYM_RTM_DMA1") {isBuiltin} : i8
//CHECK: %[[VALC5:.*]] = arith.constant 5 : i8
//CHECK: %[[VAL65:.*]] = ELFNPU37XX.Symbol %[[VALC5]] name("VPU_NNRD_SYM_FIFO_BASE") {isBuiltin} : i8
//CHECK: %[[VALC6:.*]] = arith.constant 6 : i8
//CHECK: %[[VAL66:.*]] = ELFNPU37XX.Symbol %[[VALC6]] name("VPU_NNRD_SYM_BARRIERS_START") {isBuiltin} : i8
//CHECK: %[[VAL67:.*]] = ELFNPU37XX.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags("SHF_NONE") {isBuiltin} -> !ELFNPU37XX.Section {
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL60]] : !ELFNPU37XX.Symbol
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL61]] : !ELFNPU37XX.Symbol
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL62]] : !ELFNPU37XX.Symbol
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL63]] : !ELFNPU37XX.Symbol
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL64]] : !ELFNPU37XX.Symbol
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL65]] : !ELFNPU37XX.Symbol
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL66]] : !ELFNPU37XX.Symbol

//CHECK: %[[rltDmaNetIn0:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetInput0") sourceSymbolTableSection(%[[symTabSecIn]]) targetSection(%[[VAL7]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELFNPU37XX.Section {
//CHECK: ELFNPU37XX.Reloc offset(16) <R_VPU_64> %[[symIn0]] 0
//CHECK: ELFNPU37XX.Reloc offset(144) <R_VPU_64> %[[symIn0]] 0

//CHECK: %[[rltDmaNetOut0:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetOutput0") sourceSymbolTableSection(%[[symTabSecOut]]) targetSection(%[[VAL7]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELFNPU37XX.Section {
//CHECK: ELFNPU37XX.Reloc offset(280) <R_VPU_64> %[[symOut0]] 0

//CHECK: %[[rltDmaTasks0:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.dmaTasks0") sourceSymbolTableSection(%[[VAL67]]) targetSection(%[[VAL7]]) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
//CHECK: ELFNPU37XX.Reloc offset(24) <R_VPU_64> %[[VAL60]] 0
//CHECK: ELFNPU37XX.Reloc offset(0) <R_VPU_32_RTM> %[[VAL63]] 128
//CHECK: ELFNPU37XX.Reloc offset(152) <R_VPU_64> %[[VAL60]] 0
//CHECK: ELFNPU37XX.Reloc offset(128) <R_VPU_32_RTM> %[[VAL63]] 128
//CHECK: ELFNPU37XX.Reloc offset(272) <R_VPU_64> %[[VAL60]] 0

//CHECK: %[[rltDmaNetIn1:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetInput1") sourceSymbolTableSection(%[[symTabSecIn]]) targetSection(%[[VAL12]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELFNPU37XX.Section {
//CHECK: ELFNPU37XX.Reloc offset(16) <R_VPU_64> %[[symIn0]] 0
//CHECK: ELFNPU37XX.Reloc offset(144) <R_VPU_64> %[[symIn0]] 0

//CHECK: %[[rltDmaNetOut1:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetOutput1") sourceSymbolTableSection(%[[symTabSecOut]]) targetSection(%[[VAL12]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELFNPU37XX.Section {
//CHECK: ELFNPU37XX.Reloc offset(280) <R_VPU_64> %[[symOut1]] 0

//CHECK: %[[rltDmaTasks1:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.dmaTasks1") sourceSymbolTableSection(%[[VAL67]]) targetSection(%[[VAL12]]) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
//CHECK: ELFNPU37XX.Reloc offset(24) <R_VPU_64> %[[VAL60]] 2097152
//CHECK: ELFNPU37XX.Reloc offset(0) <R_VPU_32_RTM> %[[VAL64]] 128
//CHECK: ELFNPU37XX.Reloc offset(152) <R_VPU_64> %[[VAL60]] 2097152
//CHECK: ELFNPU37XX.Reloc offset(128) <R_VPU_32_RTM> %[[VAL64]] 128
//CHECK: ELFNPU37XX.Reloc offset(272) <R_VPU_64> %[[VAL60]] 2097152

//CHECK: %[[rltMi:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%[[symTabTasks]]) targetSection(%[[VAL14]]) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
//CHECK: ELFNPU37XX.Reloc offset(72) <R_VPU_64> %[[symDmaSec0]] 0
//CHECK: ELFNPU37XX.Reloc offset(112) <R_VPU_64> %[[symDmaSec1]] 0
//CHECK: ELFNPU37XX.Reloc offset(312) <R_VPU_64> %[[symBarSec]] 0
