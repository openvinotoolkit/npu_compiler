//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --deduce-dynamic-mi %s | FileCheck %s
// REQUIRES: dev-build && arch-NPU50XX

module @setDynMi {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main() {
    ELF.Main @ELFMain {
      ELF.CreateSection @note.MappedInferenceVersion aligned(64) secType(SHT_NOTE) secFlags("SHF_NONE") secLocation(<DDR>) {
        NPUReg50XX.MappedInferenceVersion(11 _ 5 _ 0) {sym_name = "MappedInferenceVersion_0_0"}
      }
      ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") secLocation(<DDR>) {
          NPUReg50XX.ManagedMappedInference descriptor = <
          VpuManagedMappedInference {
            MMI_vpu_nnrt_api_ver = UINT 0,
            MMI_final_barrier = UINT 10,
            taskReferenceR1_MMI_work_item = UINT 0,
            taskReferenceR2_MMI_work_item = UINT 0,
            taskReferenceR3_MMI_work_item = UINT 0,
            taskReferenceAddr_MMI_work_item = UINT 0,
            taskReferenceCount_MMI_work_item = UINT 0,
            taskReferenceR1_MMI_task_configs = UINT 0,
            taskReferenceR2_MMI_task_configs = UINT 0,
            taskReferenceR3_MMI_task_configs = UINT 0,
            taskReferenceAddr_MMI_task_configs = UINT 0,
            taskReferenceCount_MMI_task_configs = UINT 0,
            taskReferenceR1_MMI_reserved0_0 = UINT 0,
            taskReferenceR2_MMI_reserved0_0 = UINT 0,
            taskReferenceR3_MMI_reserved0_0 = UINT 0,
            taskReferenceAddr_MMI_reserved0_0 = UINT 0,
            taskReferenceCount_MMI_reserved0_0 = UINT 0,
            taskReferenceR1_MMI_reserved0_1 = UINT 0,
            taskReferenceR2_MMI_reserved0_1 = UINT 0,
            taskReferenceR3_MMI_reserved0_1 = UINT 0,
            taskReferenceAddr_MMI_reserved0_1 = UINT 0,
            taskReferenceCount_MMI_reserved0_1 = UINT 0,
            taskReferenceR1_MMI_reserved0_2 = UINT 0,
            taskReferenceR2_MMI_reserved0_2 = UINT 0,
            taskReferenceR3_MMI_reserved0_2 = UINT 0,
            taskReferenceAddr_MMI_reserved0_2 = UINT 0,
            taskReferenceCount_MMI_reserved0_2 = UINT 0,
            taskReferenceR1_MMI_reserved0_3 = UINT 0,
            taskReferenceR2_MMI_reserved0_3 = UINT 0,
            taskReferenceR3_MMI_reserved0_3 = UINT 0,
            taskReferenceAddr_MMI_reserved0_3 = UINT 0,
            taskReferenceCount_MMI_reserved0_3 = UINT 0,
            taskReferenceR1_MMI_barriers_configuration = UINT 0,
            taskReferenceR2_MMI_barriers_configuration = UINT 0,
            taskReferenceR3_MMI_barriers_configuration = UINT 0,
            taskReferenceAddr_MMI_barriers_configuration = UINT 0,
            taskReferenceCount_MMI_barriers_configuration = UINT 87,
            taskReferenceR1_MMI_num_of_barrier_reprogrammings = UINT 0,
            taskReferenceR2_MMI_num_of_barrier_reprogrammings = UINT 0,
            taskReferenceR3_MMI_num_of_barrier_reprogrammings = UINT 0,
            taskReferenceAddr_MMI_num_of_barrier_reprogrammings = UINT 0,
            taskReferenceCount_MMI_num_of_barrier_reprogrammings = UINT 0,
            taskReferenceR1_MMI_initial_barriers = UINT 0,
            taskReferenceR2_MMI_initial_barriers = UINT 0,
            taskReferenceR3_MMI_initial_barriers = UINT 0,
            taskReferenceAddr_MMI_initial_barriers = UINT 0,
            taskReferenceCount_MMI_initial_barriers = UINT 0,
            taskReferenceR1_MMI_nnrt_config = UINT 0,
            taskReferenceR2_MMI_nnrt_config = UINT 0,
            taskReferenceR3_MMI_nnrt_config = UINT 0,
            taskReferenceAddr_MMI_nnrt_config = UINT 0,
            taskReferenceCount_MMI_nnrt_config = UINT 0,
            MMI_actshv_used = UINT 0,
            MMI_dpu_used = UINT 0,
            MMI_media_used = UINT 0,
            MMI_dma_from_ddr_used = UINT 0,
            MMI_dma_from_cmx_used = UINT 0,
            MMI_pad0_ = UINT 0,
            MMI_barrier_programming_mode = UINT 0,
            taskReferenceR1_MMI_inference_info = UINT 0,
            taskReferenceR2_MMI_inference_info = UINT 0,
            taskReferenceR3_MMI_inference_info = UINT 0,
            taskReferenceAddr_MMI_inference_info = UINT 0,
            taskReferenceCount_MMI_inference_info = UINT 0,
            MMI_barrier_configuration_stride = UINT 0,
            MMI_inference_feature_cfg {
              UINT MMI_DisableDmaSwFifo = 0,
              UINT MMI_ReservedInferenceConfig = 0,
            }
            MMI_pad1_ = UINT 0,
            MMI_model_identifier = UINT 1,
            MMI_pad2_0 = UINT 0,
            MMI_pad2_1 = UINT 0,
            MMI_pad2_2 = UINT 0,
            MMI_pad2_3 = UINT 0,
            MMI_pad2_4 = UINT 0,
            MMI_pad2_5 = UINT 0,
            MMI_pad2_6 = UINT 0,
            MMI_pad2_7 = UINT 0,
            MMI_pad2_8 = UINT 0,
            MMI_pad2_9 = UINT 0,
            MMI_pad2_10 = UINT 0,
            MMI_pad2_11 = UINT 0,
            MMI_pad2_12 = UINT 0,
            MMI_pad2_13 = UINT 0,
            MMI_pad2_14 = UINT 0,
            MMI_pad2_15 = UINT 0,
            MMI_pad2_16 = UINT 0,
            MMI_pad2_17 = UINT 0,
            MMI_pad2_18 = UINT 0,
            MMI_pad2_19 = UINT 0,
            MMI_pad2_20 = UINT 0,
            MMI_pad2_21 = UINT 0,
            MMI_pad2_22 = UINT 0,
            MMI_pad2_23 = UINT 0,
            MMI_pad2_24 = UINT 0,
            MMI_pad2_25 = UINT 0,
            MMI_pad2_26 = UINT 0,
            MMI_pad2_27 = UINT 0,
            MMI_pad2_28 = UINT 0,
            MMI_bootstrap_workitems_count = UINT 0,
            MMI_pad3_ = UINT 0,
          } requires 11:10:0
        > {barrierTasks = @program.managedBarrier::@ConfigureBarrier_0_0, bootstrapBarriers = @program.bootstrap::@Bootstrap_0_0, dmaTasks = [[], []],
        mappedInferenceVersion = @note.MappedInferenceVersion::@MappedInferenceVersion_0_0, nnrtConfig = @program.nnrt_config::@MappedInference_nnrtConfigManaged, sym_name = "MappedInference_managed", workItems = @program.workItem::@ed.Enqueue_0_0}
      }

    }
    return
  }

}
//CHECK: ELF.CreateSection @note.MappedInferenceVersion
//CHECK:       NPUReg50XX.MappedInferenceVersion(11 _ 13 _ 0)
//CHECK: ELF.CreateSection @program.mapped_inference
//CHECK: NPUReg50XX.ManagedMappedInference descriptor = <
//CHECK: VpuManagedMappedInference
//CHECK:             MMI_vpu_nnrt_api_ver = UINT 0xB000D
//CHECK: requires 11:13:0
