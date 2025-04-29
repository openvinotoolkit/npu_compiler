//
// Copyright (C) 2021-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// VPU translate expectedly failed dueto unsupported DDR memory for actshave
// RUN: vpux-opt --init-compiler="vpu-arch=%arch% allow-custom-values=true" --lower-VPUIP-to-ELF %s -o %t.mlir
// RUN: vpux-translate --vpu-arch=%arch% --export-ELF %t.mlir -o %t.elf
// RUN: vpux-translate --vpu-arch=%arch% --import-ELF %t.elf | FileCheck %s
// RUN: rm %t.elf %t.mlir
// REQUIRES: arch-NPU37XX

module @Test {

IE.TileResource 1 of @NCE

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1000xf32>
    }
    outputsInfo : {
        DataInfo "softmax" : tensor<1x1000xf32>
    }

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func private @builtin_Softmax(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    %1 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    VPURT.Task updates(%1 : !VPURT.Barrier) {
        %2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_Softmax inputs(%arg0 as %arg2: memref<1x1x1x1000xf16>) outputs(%0 as %arg3: memref<1x1x1x1000xf16, @DDR>) on tile 0 -> memref<1x1x1x1000xf16, @DDR>  {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3) : memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16, @DDR>
            }
    }
    VPURT.Task waits(%1 : !VPURT.Barrier) {
        %2 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x1x1x1000xf16, @DDR>) outputs(%arg1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    }
    return %arg1: memref<1x1x1x1000xf16>
}

}

// CHECK: %[[VAL0:.*]] =  VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, 4294967295> -> !VPURegMapped.Index<0:0:0>
// CHECK: %[[VAL1:.*]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.BarrierConfigs"} -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL0]] : !VPURegMapped.Index<0:0:0>

// CHECK: %[[VAL2:.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<2000xui8, @DDR>
// CHECK: %[[VAL3:.*]] = VPUMI37XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 2000 : i64, srcWidth = 2000 : i64, srcStride = 2000 : i64, srcPlaneStride = 0 : i64, dstWidth = 2000 : i64, dstStride = 2000 : i64, dstPlaneStride = 0 : i64>, is_critical, is_out_of_order, port = 0 : si64} inputs(%[[VAL2]] : memref<2000xui8, @DDR>) outputs(%arg1 : memref<2000xui8>) waits(%[[VAL0]] : !VPURegMapped.Index<0:0:0>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
// CHECK: %[[VAL4:.*]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks0"} -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL3]] : !VPURegMapped.Index<0:0:0>

// CHECK: %[[VAL5:.*]] = VPUMI37XX.DeclareKernelText kernel_path("softmax.3720xx.elf") -> !VPURegMapped.Index<0:0:0>
// CHECK: %[[VAL6:.*]] = VPUMI37XX.DeclareKernelArgs kernel_path("softmax.3720xx.elf") -> !VPURegMapped.Index<0:0:0>
// CHECK: %[[VAL7:.*]] = VPUMI37XX.DeclareKernelEntry kernel_path("softmax.3720xx.elf") -> !VPURegMapped.Index<0:0:0>
// CHECK: %[[VAL8:.*]] = VPUMI37XX.ActKernelRange kernel_text_index(%[[VAL5]] : !VPURegMapped.Index<0:0:0>) kernel_args_index(%[[VAL6]] : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%[[VAL7]] : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
// CHECK: %[[VAL9:.*]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.ActKernelRanges"} -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL8]] : !VPURegMapped.Index<0:0:0>

// CHECK: %[[VAL10:.*]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".text.KernelText"} -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL5]] : !VPURegMapped.Index<0:0:0>

// CHECK: %[[VAL11:.*]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_WRITE|VPU_SHF_PROC_SHAVE") {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".text.KernelData"} -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL6]] : !VPURegMapped.Index<0:0:0>

// CHECK: %[[VAL12:.*]] = VPUMI37XX.ActKernelInvocation range_index(%[[VAL8]] : <0:0:0>) updates(%[[VAL0]] : !VPURegMapped.Index<0:0:0>) tile(0) start_after(1) clean_after(0) -> !VPURegMapped.Index<0:0:0>
// CHECK: %[[VAL13:.*]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.ActKernelInvocations"} -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL12]] : !VPURegMapped.Index<0:0:0>

// CHECK: %[[VAL14:.*]] = VPUMI37XX.KernelParams inputs(%[[VAL2]] : memref<2000xui8, @DDR>) outputs(%[[VAL2]] : memref<2000xui8, @DDR>) dynamicInputShapes(() : ()) dynamicOutputShapes(() : ()) kernel_type("Softmax") kernel_params(dense<[0, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0]> : vector<76xui8>) -> !VPURegMapped.Index<0:0:0>
// CHECK: %[[VAL15:.*]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".text.KernelParams"} -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL14]] : !VPURegMapped.Index<0:0:0>

// CHECK: %[[VAL16:.*]] = VPUMI37XX.ActShaveRt kernel("nnActEntry") -> !VPURegMapped.Index<0:0:0>
// CHECK: %[[VAL17:.*]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".text.actKernelRtConfigSec"} -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL16]] : !VPURegMapped.Index<0:0:0>

// CHECK: %[[VAL18:.*]] = VPUMI37XX.MappedInference dmas(%[[VAL3]] : !VPURegMapped.Index<0:0:0>) actKernelRanges(%[[VAL8]] : !VPURegMapped.Index<0:0:0>) actKernelInvocations(%[[VAL12]] : !VPURegMapped.Index<0:0:0>) barriers(%[[VAL0]] : !VPURegMapped.Index<0:0:0>) actShaveRt(%[[VAL16]] : !VPURegMapped.Index<0:0:0>) dmaCount([1]) invariantCount(0) variantCount(0) actKernelRangesCount(1) actKernelInvocationsCount(1) barrierCount(1) -> !VPURegMapped.Index<0:0:0>
// CHECK: %[[VAL19:.*]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.MappedInference"} -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL18]] : !VPURegMapped.Index<0:0:0>

// CHECK: %[[VAL20:.*]] = ELFNPU37XX.CreateMetadataSection secFlags("SHF_NONE") {secAddrAlign = 8 : i64, secInfo = 0 : i64, secName = ".metadata"} -> !ELFNPU37XX.Section {

// CHECK: %[[VAL21:.*]] = ELFNPU37XX.CreateLogicalSection secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_PROC_DMA|VPU_SHF_PROC_SHAVE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".data.BuffersIO"} -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL2]] : memref<2000xui8, @DDR>

// CHECK: %[[VAL22:.*]] = ELFNPU37XX.Symbol %[[VAL17]] name("sym_actKernelRtConfigsSec") type(<STT_NOTYPE>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
// CHECK: %[[VAL23:.*]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.actKernelRtConfig") secFlags("SHF_NONE") -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL22]] : !ELFNPU37XX.Symbol

// CHECK: %[[VAL24:.*]] = ELFNPU37XX.Symbol %arg0 name("input") type(<STT_NOTYPE>) size(2000) {value = 0 : ui64} : memref<2000xui8>
// CHECK: %[[VAL25:.*]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL24]] : !ELFNPU37XX.Symbol

// CHECK: %[[VAL26:.*]] = ELFNPU37XX.Symbol %arg1 name("softmax") type(<STT_NOTYPE>) size(2000) {value = 0 : ui64} : memref<2000xui8>
// CHECK: %[[VAL27:.*]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL26]] : !ELFNPU37XX.Symbol

// CHECK: %[[VAL28:.*]] = ELFNPU37XX.Symbol %[[VAL21]] name("sym_bufferSection") type(<STT_NOTYPE>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
// CHECK: %[[VAL29:.*]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.buffers") secFlags("SHF_NONE") -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL28]] : !ELFNPU37XX.Symbol

// CHECK: %[[VAL30:.*]] = ELFNPU37XX.Symbol %[[VAL4]] name("sym_dmaSection0") type(<STT_NOTYPE>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
// CHECK: %[[VAL31:.*]] = ELFNPU37XX.Symbol %[[VAL1]] name("sym_barrierSection") type(<STT_NOTYPE>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
// CHECK: %[[VAL32:.*]] = ELFNPU37XX.Symbol %[[VAL10]] name("sym_kernelTextSection") type(<STT_NOTYPE>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
// CHECK: %[[VAL33:.*]] = ELFNPU37XX.Symbol %[[VAL11]] name("sym_kernelDataSection") type(<STT_NOTYPE>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
// CHECK: %[[VAL34:.*]] = ELFNPU37XX.Symbol %[[VAL15]] name("sym_kernelParamsSection") type(<STT_NOTYPE>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
// CHECK: %[[VAL35:.*]] = ELFNPU37XX.Symbol %[[VAL9]] name("sym_actKernelRangeSection") type(<STT_NOTYPE>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
// CHECK: %[[VAL36:.*]] = ELFNPU37XX.Symbol %[[VAL13]] name("sym_actKernelInvo") type(<STT_NOTYPE>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
// CHECK: %[[VAL37:.*]] = ELFNPU37XX.Symbol %[[VAL19]] name("MappedInference_entry") type(<VPU_STT_ENTRY>) size(0) {value = 0 : ui64} : !ELFNPU37XX.Section
// CHECK: %[[VAL38:.*]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.tasks") secFlags("SHF_NONE") -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL30]] : !ELFNPU37XX.Symbol
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL31]] : !ELFNPU37XX.Symbol
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL32]] : !ELFNPU37XX.Symbol
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL33]] : !ELFNPU37XX.Symbol
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL34]] : !ELFNPU37XX.Symbol
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL35]] : !ELFNPU37XX.Symbol
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL36]] : !ELFNPU37XX.Symbol
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL37]] : !ELFNPU37XX.Symbol

// CHECK: %[[VALC0:.*]] = arith.constant 0 : i8
// CHECK: %[[VAL39:.*]] = ELFNPU37XX.Symbol %[[VALC0]] name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") {isBuiltin} : i8
// CHECK: %[[VALC1:.*]] = arith.constant 1 : i8
// CHECK: %[[VAL40:.*]] = ELFNPU37XX.Symbol %[[VALC1]] name("VPU_NNRD_SYM_RTM_IVAR") {isBuiltin} : i8
// CHECK: %[[VALC2:.*]] = arith.constant 2 : i8
// CHECK: %[[VAL41:.*]] = ELFNPU37XX.Symbol %[[VALC2]] name("VPU_NNRD_SYM_RTM_ACT") {isBuiltin} : i8
// CHECK: %[[VALC3:.*]] = arith.constant 3 : i8
// CHECK: %[[VAL42:.*]] = ELFNPU37XX.Symbol %[[VALC3]] name("VPU_NNRD_SYM_RTM_DMA0") {isBuiltin} : i8
// CHECK: %[[VALC4:.*]] = arith.constant 4 : i8
// CHECK: %[[VAL43:.*]] = ELFNPU37XX.Symbol %[[VALC4]] name("VPU_NNRD_SYM_RTM_DMA1") {isBuiltin} : i8
// CHECK: %[[VALC5:.*]] = arith.constant 5 : i8
// CHECK: %[[VAL44:.*]] = ELFNPU37XX.Symbol %[[VALC5]] name("VPU_NNRD_SYM_FIFO_BASE") {isBuiltin} : i8
// CHECK: %[[VALC6:.*]] = arith.constant 6 : i8
// CHECK: %[[VAL45:.*]] = ELFNPU37XX.Symbol %[[VALC6]] name("VPU_NNRD_SYM_BARRIERS_START") {isBuiltin} : i8
// CHECK: %[[VALC7:.*]] = arith.constant 7 : i8
// CHECK: %[[VAL46:.*]] = ELFNPU37XX.Symbol %[[VALC7]] name("VPU_NNRD_SYM_HW_REGISTER") {isBuiltin} : i8
// CHECK: %[[VAL47:.*]] = ELFNPU37XX.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags("SHF_NONE") {isBuiltin} -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL39]] : !ELFNPU37XX.Symbol
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL40]] : !ELFNPU37XX.Symbol
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL41]] : !ELFNPU37XX.Symbol
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL42]] : !ELFNPU37XX.Symbol
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL43]] : !ELFNPU37XX.Symbol
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL44]] : !ELFNPU37XX.Symbol
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL45]] : !ELFNPU37XX.Symbol
// CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL46]] : !ELFNPU37XX.Symbol

// CHECK: %[[VAL48:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.DMA_NetOutput0") sourceSymbolTableSection(%[[VAL27]]) targetSection(%[[VAL4]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.Reloc offset(24) <R_VPU_64> %[[VAL26]] 0 {description = ""}

// CHECK: %[[VAL49:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.dmaTasks0") sourceSymbolTableSection(%[[VAL29]]) targetSection(%[[VAL4]]) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.Reloc offset(16) <R_VPU_64> %[[VAL28]] 0 {description = ""}

// CHECK: %[[VAL50:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection(%[[VAL38]]) targetSection(%[[VAL15]]) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.Reloc offset(12) <R_VPU_32> %[[VAL34]] 80 {description = ""}
// CHECK-NEXT: ELFNPU37XX.Reloc offset(16) <R_VPU_32> %[[VAL34]] 96 {description = ""}
// CHECK-NEXT: ELFNPU37XX.Reloc offset(48) <R_VPU_32> %[[VAL34]] 128 {description = ""}
// CHECK-NEXT: ELFNPU37XX.Reloc offset(52) <R_VPU_32> %[[VAL34]] 144 {description = ""}

// CHECK: %[[VAL51:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.Kernel_NetInput0") sourceSymbolTableSection(%[[VAL25]]) targetSection(%[[VAL15]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.Reloc offset(0) <R_VPU_32> %[[VAL24]] 0 {description = ""}

// CHECK: %[[VAL52:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection(%[[VAL29]]) targetSection(%[[VAL15]]) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.Reloc offset(36) <R_VPU_32> %[[VAL28]] 0 {description = ""}

// CHECK: %[[VAL53:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.ActKernelRanges") sourceSymbolTableSection(%[[VAL38]]) targetSection(%[[VAL9]]) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.Reloc offset(8) <R_VPU_32> %[[VAL32]] 0 {description = ""}

// CHECK: %[[VAL54:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.ActKernelInvocations") sourceSymbolTableSection(%[[VAL47]]) targetSection(%[[VAL13]]) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.Reloc offset(0) <R_VPU_32_RTM> %[[VAL41]] 24 {description = ""}

// CHECK: %[[VAL55:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.ActKernelInvocations") sourceSymbolTableSection(%[[VAL38]]) targetSection(%[[VAL13]]) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.Reloc offset(8) <R_VPU_32> %[[VAL33]] 0 {description = ""}
// CHECK-NEXT: ELFNPU37XX.Reloc offset(4) <R_VPU_32> %[[VAL34]] 0 {description = ""}

// CHECK: %[[VAL55:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%[[VAL23]]) targetSection(%[[VAL19]]) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.Reloc offset(340) <R_VPU_32> %[[VAL22]] 0 {description = ""}

// CHECK: %[[VAL56:.*]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%[[VAL38]]) targetSection(%[[VAL19]]) secFlags(SHF_INFO_LINK) -> !ELFNPU37XX.Section {
// CHECK-NEXT: ELFNPU37XX.Reloc offset(72) <R_VPU_64> %[[VAL30]] 0 {description = ""}
// CHECK-NEXT: ELFNPU37XX.Reloc offset(312) <R_VPU_64> %[[VAL31]] 0 {description = ""}
// CHECK-NEXT: ELFNPU37XX.Reloc offset(232) <R_VPU_64> %[[VAL35]] 0 {description = ""}
// CHECK-NEXT: ELFNPU37XX.Reloc offset(272) <R_VPU_64> %[[VAL36]] 0 {description = ""}
