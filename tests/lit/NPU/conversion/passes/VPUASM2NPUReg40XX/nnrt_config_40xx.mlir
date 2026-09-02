//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --convert-VPUASM-to-NPUReg40XX %s | FileCheck %s
// REQUIRES: dev-build && platform-NPU4000

module @Test {
  config.ExecutorResource 1 of @M2I
  config.ExecutorResource 1 of @DMA_NN
  config.Resources 6 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main() {
    ELF.Main {
      ELF.CreateSection @shave.runtime aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.ActShaveRt @ActShaveRt kernel("nnActEntry")
      }
      ELF.CreateSection @program.nnrt_config aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") secLocation(<DDR>) {
        VPUASM.nnrtConfig {elfMemOffsetAttrKey = 0 : ui64} <{actShaveRt = @shave.runtime::@ActShaveRt, isActKernelInvocations}> @MappedInference_nnrtConfigManaged : dmaHwpBase(@buffer.CMX_NN.0::@DeclareBuffer6)
      }
    }
    return
  }
}

//CHECK: NPUReg40XX.NNrtConfig <{actShaveRt = @shave.runtime::@ActShaveRt, descriptor = #NPUReg40XX.VpuNNRTConfig<
//CHECK:   VpuNNRTConfig {
//CHECK:     NNRTCfg_reserved = UINT 0,
//CHECK:     NNRTCfg_runtime_entry = UINT 0x1C0009A0,
//CHECK:     NNRTCfg_act_rt_window_base = UINT 0,
//CHECK:     NNRTCfg_stack_0 = UINT 0,
//CHECK:     NNRTCfg_stack_1 = UINT 0,
//CHECK:     NNRTCfg_stack_2 = UINT 0,
//CHECK:     NNRTCfg_stack_3 = UINT 0,
//CHECK:     NNRTCfg_stack_4 = UINT 0,
//CHECK:     NNRTCfg_stack_5 = UINT 0,
//CHECK:     NNRTCfg_stack_6 = UINT 0,
//CHECK:     NNRTCfg_stack_7 = UINT 0,
//CHECK:     NNRTCfg_stack_8 = UINT 0,
//CHECK:     NNRTCfg_stack_9 = UINT 0,
//CHECK:     NNRTCfg_stack_10 = UINT 0,
//CHECK:     NNRTCfg_stack_11 = UINT 0,
//CHECK:     NNRTCfg_stack_size = UINT 0,
//CHECK:     NNRTCfg_deprecated1 = UINT 0,
//CHECK:     NNRTCfg_perf_metrics_mask = UINT 0,
//CHECK:     NNRTCfg_runtime_version = UINT 0x10009,
//CHECK:     NNRTCfg_use_schedule_embedded_rt = UINT 1,
//CHECK:     NNRTCfg_deprecated3 = UINT 0,
//CHECK:     NNRTCfg_pad_6 = UINT 0,
//CHECK:     NNRTCfg_logAddrDmaHwp = UINT 0,
//CHECK:     NNRTCfg_HwpCfgAddr = UINT 0,
//CHECK:   } requires 11:4:10
//CHECK: >, dmaHwpBase = @buffer.CMX_NN.0::@DeclareBuffer6, isActKernelInvocations, sym_name = "MappedInference_nnrtConfigManaged"}>
