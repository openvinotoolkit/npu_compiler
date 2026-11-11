//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --add-host-parsed-inference %s | FileCheck %s
// REQUIRES: platform-NPU6010

net.NetworkInfo entryPoint : @oneDma inputsInfo : {
  DataInfo "input" : tensor<1x2x3x4xf16>
} outputsInfo : {
  DataInfo "output" : tensor<1x2x3x4xf16>
}
func.func @oneDma() {
  ELF.Main {
    ELF.CreateSection @note.MappedInferenceVersion aligned(4) secType(SHT_NOTE) secFlags("SHF_NONE") secLocation(<DDR>) {
      VPUASM.MappedInferenceVersion @MappedInferenceVersion_0_0(11 _ 4 _ 10)
    }

    ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
      VPUASM.ManagedMappedInference @MappedInference_managed : dmas () workItems(@program.workItem::@Enqueue_0_0_0) barrierTasks(@program.managedBarrier::@ConfigureBarrier_0_0_0) bootstrapBarriers() nnrtConfig(@program.nnrt_config::@MappedInference_nnrtConfigManaged) mappedInferenceVersion(@note.MappedInferenceVersion::@MappedInferenceVersion_0_0_0) {actshv_used = 0 : ui8, barrierConfigurationStride = 0 : i64, barrierConfigurationTasksCount = 0 : i64, barrierCount = 0 : i64, barriersReprogrammingCount = 0 : i64, bootstrapWorkItemsCount = 1 : i64, bootstrapBarriersCount = 0 : i64, disableDmaSwFifo = true, dmaCount = [[0, 0], [0, 0]], dma_from_cmx_used = 0 : ui8, dma_from_ddr_used = 1 : ui8, dpu_used = 0 : ui8, final_barrier_id = 0 : i64, workItemsCount = 1 : i64, workloadManagementBarrierProgrammingMode = #VPURegMapped.workload_management_barrier_programming_mode<ALL_BARRIER_DMAS_SCHEDULED>}
    }
  }
  return
}

// CHECK-LABEL: func.func @oneDma
// CHECK: ELF.Main {
// CHECK: ELF.CreateSection @program.host_parsed_inference
// CHECK-SAME: secType(VPU_SHT_HPI)
// CHECK: VPUASM.HostParsedInference @HostParsedInference
// CHECK-SAME: mapped(@program.mapped_inference::@MappedInference_managed)
