//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --convert-VPUASM-to-NPUReg40XX %s | FileCheck %s
// REQUIRES: dev-build && arch-NPU40XX

module @OneDMAWithoutAttributes attributes {config.arch = #config.arch_kind<NPU40XX>} {
  config.ExecutorResource 1 of @M2I
  config.ExecutorResource 1 of @DMA_NN
  config.Resources 6 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main() {
    ELF.Main @ELFMain {
      VPUASM.ConfigureBarrier @ConfigureBarrier_0_0 idx(!VPURegMapped.Index<0:0:0>) workItemIdx(!VPURegMapped.Index<0:0:0>) (0) => (-1) counts(3 : 1)
      VPUASM.ConfigureBarrier @ConfigureBarrier_0_1 idx(!VPURegMapped.Index<0:0:1>) (17) => (12) counts(34 : 43)
      VPUASM.ManagedBarrier @ConfigureBarrier_0_2 idx(!VPURegMapped.Index<0:0:2>) workItemIdx(!VPURegMapped.Index<0:0:999>) (0) => (-1) counts(4 : 5)
      VPUASM.ManagedBarrier @ConfigureBarrier_0_3 idx(!VPURegMapped.Index<0:0:3>) (1) => (4) counts(23 : 32)
    }
      return
  }
}

//CHECK-LABEL: @main
//CHECK:       NPUReg40XX.ConfigureBarrier descriptor = <
//CHECK:         VpuBarrierCountConfig {
//CHECK:           next_same_id_ = UINT 0xFFFFFFFF,
//CHECK:           producer_count_ = UINT 3,
//CHECK:           consumer_count_ = UINT 1,
//CHECK:           real_id_ = UINT 0,
//CHECK:           barcfg_pad_3_ = UINT 0
//CHECK:         } requires 11:4:10
//CHECK:       > {sym_name = "ConfigureBarrier_0_0"}
//CHECK:       NPUReg40XX.ConfigureBarrier descriptor = <
//CHECK:         VpuBarrierCountConfig {
//CHECK:           next_same_id_ = UINT 0xC,
//CHECK:           producer_count_ = UINT 0x22,
//CHECK:           consumer_count_ = UINT 0x2B,
//CHECK:           real_id_ = UINT 0x11,
//CHECK:           barcfg_pad_3_ = UINT 0
//CHECK:         } requires 11:4:10
//CHECK:       > {sym_name = "ConfigureBarrier_0_1"}
//CHECK:       NPUReg40XX.ManagedBarrier descriptor = <
//CHECK:         VpuTaskBarrierMap {
//CHECK:           tb_next_same_id = UINT 0xFFFFFFFF,
//CHECK:           tb_producer_count = UINT 4,
//CHECK:           tb_consumer_count = UINT 5,
//CHECK:           tb_real_id = UINT 0,
//CHECK:           tb_pad3 = UINT 0,
//CHECK:           tb_work_item_idx = UINT 0x3E7,
//CHECK:           tb_enqueue_count = UINT 0,
//CHECK:           tb_reserved_next_enqueue_id = UINT 0
//CHECK:         } requires 11:4:10
//CHECK:       > {sym_name = "ConfigureBarrier_0_2"}
//CHECK:       NPUReg40XX.ManagedBarrier descriptor = <
//CHECK:         VpuTaskBarrierMap {
//CHECK:           tb_next_same_id = UINT 4,
//CHECK:           tb_producer_count = UINT 0x17,
//CHECK:           tb_consumer_count = UINT 0x20,
//CHECK:           tb_real_id = UINT 1,
//CHECK:           tb_pad3 = UINT 0,
//CHECK:           tb_work_item_idx = UINT 0,
//CHECK:           tb_enqueue_count = UINT 0,
//CHECK:           tb_reserved_next_enqueue_id = UINT 0
//CHECK:         } requires 11:4:10
//CHECK:       > {sym_name = "ConfigureBarrier_0_3"}
