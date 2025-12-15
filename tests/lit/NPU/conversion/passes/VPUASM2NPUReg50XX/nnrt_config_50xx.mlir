//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --convert-VPUASM-to-NPUReg50XX %s | FileCheck %s
// REQUIRES: dev-build && arch-NPU50XX

module @Test attributes {config.arch = #config.arch_kind<NPU50XX>} {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main() {
    ELF.Main @ELFMain {
      ELF.CreateSection @shave.runtime aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.ActShaveRt @ActShaveRt kernel("nnActEntry")
      }
      ELF.CreateSection @program.nnrt_config aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") secLocation(<DDR>) {
        VPUASM.nnrtConfig {actShaveRt = @shave.runtime::@ActShaveRt, elfMemOffsetAttrKey = 0 : ui64, isActKernelInvocations} @MappedInference_nnrtConfigManaged : dmaHwpBase(@buffer.CMX_NN.0::@DeclareBuffer6)
      }
    }
    return
  }
}

//CHECK: NPUReg50XX.NNrtConfig descriptor = <
//CHECK:   VpuNNRTConfig {
//CHECK:     NNRTCfg_reserved = UINT 0,
//CHECK:     NNRTCfg_runtime_entry = UINT 0x1C000000,
//CHECK:     NNRTCfg_act_rt_window_base = UINT 0,
//CHECK:     NNRTCfg_stack_0 = UINT 0x40001C00,
//CHECK:     NNRTCfg_stack_1 = UINT 0x40003800,
//CHECK:     NNRTCfg_stack_2 = UINT 0x40001C00,
//CHECK:     NNRTCfg_stack_3 = UINT 0x40003800,
//CHECK:     NNRTCfg_stack_4 = UINT 0x40001C00,
//CHECK:     NNRTCfg_stack_5 = UINT 0x40003800,
//CHECK:     NNRTCfg_stack_6 = UINT 0,
//CHECK:     NNRTCfg_stack_7 = UINT 0,
//CHECK:     NNRTCfg_stack_8 = UINT 0,
//CHECK:     NNRTCfg_stack_9 = UINT 0,
//CHECK:     NNRTCfg_stack_10 = UINT 0,
//CHECK:     NNRTCfg_stack_11 = UINT 0,
//CHECK:     NNRTCfg_stack_size = UINT 0x1C00,
//CHECK:     NNRTCfg_code_window_buffer_size = UINT 0x2630,
//CHECK:     NNRTCfg_perf_metrics_mask = UINT 0,
//CHECK:     NNRTCfg_runtime_version = UINT 0x1000F,
//CHECK:     NNRTCfg_use_schedule_embedded_rt = UINT 1,
//CHECK:     NNRTCfg_dpu_perf_mode = UINT 3,
//CHECK:     NNRTCfg_pad_6 = UINT 0,
//CHECK:     NNRTCfg_logAddrDmaHwp = UINT 0,
//CHECK:     NNRTCfg_HwpCfgAddr = UINT 0,
//CHECK:   } requires 11:13:0
//CHECK: > {actShaveRt = @shave.runtime::@ActShaveRt, dmaHwpBase = @buffer.CMX_NN.0::@DeclareBuffer6, isActKernelInvocations, sym_name = "MappedInference_nnrtConfigManaged"}
