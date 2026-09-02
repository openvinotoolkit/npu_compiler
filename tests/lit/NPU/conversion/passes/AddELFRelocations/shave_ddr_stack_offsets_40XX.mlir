//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform%" --set-elf-op-offsets %s | FileCheck %s
// REQUIRES: dev-build && platform-NPU4000

func.func @setOffsets() {
  ELF.Main {
    ELF.CreateLogicalSection @shave.stackBuffer aligned(64) secType(SHT_NOBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_SHAVE") secLocation(<DDR>) {
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_0 : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_1 : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_2 : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_3 : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_4 : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_5 : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_6 : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_7 : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_8 : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_9 : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_10 : 16384
      VPUASM.ActShaveRtStack @ActShaveRtStack_0_11 : 16384
    }
    ELF.CreateSection @program.nnrt_config aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") secLocation(<DDR>) {
      NPUReg40XX.NNrtConfig <{descriptor = #NPUReg40XX.VpuNNRTConfig<
          VpuNNRTConfig {
            NNRTCfg_reserved = UINT 0,
            NNRTCfg_runtime_entry = UINT 0x1C000A60,
            NNRTCfg_act_rt_window_base = UINT 0,
            NNRTCfg_stack_0 = UINT 0,
            NNRTCfg_stack_1 = UINT 0,
            NNRTCfg_stack_2 = UINT 0,
            NNRTCfg_stack_3 = UINT 0,
            NNRTCfg_stack_4 = UINT 0,
            NNRTCfg_stack_5 = UINT 0,
            NNRTCfg_stack_6 = UINT 0,
            NNRTCfg_stack_7 = UINT 0,
            NNRTCfg_stack_8 = UINT 0,
            NNRTCfg_stack_9 = UINT 0,
            NNRTCfg_stack_10 = UINT 0,
            NNRTCfg_stack_11 = UINT 0,
            NNRTCfg_stack_size = UINT 0,
            NNRTCfg_deprecated1 = UINT 0,
            NNRTCfg_perf_metrics_mask = UINT 0,
            NNRTCfg_runtime_version = UINT 0x10008,
            NNRTCfg_use_schedule_embedded_rt = UINT 1,
            NNRTCfg_deprecated3 = UINT 0,
            NNRTCfg_pad_6 = UINT 0,
            NNRTCfg_logAddrDmaHwp = UINT 0,
            NNRTCfg_HwpCfgAddr = UINT 0,
          } requires 11:4:10
        >, actShaveRt = @shave.runtime::@ActShaveRt, dmaHwpBase = @buffer.CMX_NN.0::@DeclareBuffer6, isActKernelInvocations, actShaveStacks = [@shave.stackBuffer::@ActShaveRtStack_0_0,
        @shave.stackBuffer::@ActShaveRtStack_0_1, @shave.stackBuffer::@ActShaveRtStack_0_2, @shave.stackBuffer::@ActShaveRtStack_0_3, @shave.stackBuffer::@ActShaveRtStack_0_4,
        @shave.stackBuffer::@ActShaveRtStack_0_5, @shave.stackBuffer::@ActShaveRtStack_0_6, @shave.stackBuffer::@ActShaveRtStack_0_7, @shave.stackBuffer::@ActShaveRtStack_0_8,
        @shave.stackBuffer::@ActShaveRtStack_0_9, @shave.stackBuffer::@ActShaveRtStack_0_10, @shave.stackBuffer::@ActShaveRtStack_0_11], sym_name = "MappedInference_nnrtConfigManaged"}>
    }
    ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
      ELF.Symbol @elfsym.shave.stackBuffer of(@shave.stackBuffer) type(<STT_SECTION>) size(0) value(0)
      ELF.Symbol @elfsym.program.nnrt_config of(@program.nnrt_config) type(<STT_SECTION>) size(0) value(0)
    }
  }
  return
}


//CHECK:   ELF.CreateLogicalSection @shave.stackBuffer aligned(64) secType(SHT_NOBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_SHAVE") secLocation(<DDR>) {
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_0 {elfMemOffsetAttrKey = 0 : ui64} : 16384
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_1 {elfMemOffsetAttrKey = 16384 : ui64} : 16384
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_2 {elfMemOffsetAttrKey = 32768 : ui64} : 16384
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_3 {elfMemOffsetAttrKey = 49152 : ui64} : 16384
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_4 {elfMemOffsetAttrKey = 65536 : ui64} : 16384
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_5 {elfMemOffsetAttrKey = 81920 : ui64} : 16384
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_6 {elfMemOffsetAttrKey = 98304 : ui64} : 16384
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_7 {elfMemOffsetAttrKey = 114688 : ui64} : 16384
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_8 {elfMemOffsetAttrKey = 131072 : ui64} : 16384
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_9 {elfMemOffsetAttrKey = 147456 : ui64} : 16384
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_10 {elfMemOffsetAttrKey = 163840 : ui64} : 16384
//CHECK:     VPUASM.ActShaveRtStack @ActShaveRtStack_0_11 {elfMemOffsetAttrKey = 180224 : ui64} : 16384
//CHECK:   ELF.CreateSection @program.nnrt_config
//CHECK: actShaveStacks = [@shave.stackBuffer::@ActShaveRtStack_0_0,
//CHECK:        @shave.stackBuffer::@ActShaveRtStack_0_1, @shave.stackBuffer::@ActShaveRtStack_0_2, @shave.stackBuffer::@ActShaveRtStack_0_3, @shave.stackBuffer::@ActShaveRtStack_0_4,
//CHECK:        @shave.stackBuffer::@ActShaveRtStack_0_5, @shave.stackBuffer::@ActShaveRtStack_0_6, @shave.stackBuffer::@ActShaveRtStack_0_7, @shave.stackBuffer::@ActShaveRtStack_0_8,
//CHECK:        @shave.stackBuffer::@ActShaveRtStack_0_9, @shave.stackBuffer::@ActShaveRtStack_0_10, @shave.stackBuffer::@ActShaveRtStack_0_11
