//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --deduce-dynamic-mi %s | FileCheck %s
// REQUIRES: arch-NPU40XX


module @setDynMi {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main() {
    ELF.Main @ELFMain {
      ELF.CreateSection @note.MappedInferenceVersion aligned(64) secType(SHT_NOTE) secFlags("SHF_NONE") {
        NPUReg40XX.MappedInferenceVersion(11 _ 4 _ 10) {sym_name = "MappedInferenceVersion_0_0"}
      }
      ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") {
        "NPUReg40XX.MappedInference"() <{actKernelInvocations = [], actKernelInvocationsCount = [], actKernelRanges = [], actKernelRangesCount = [], barrierCount = 0 : i64, dmaCMXCount = [0, 0], dmaCount = [[0, 0], [0, 0]], dmaDDRCount = [0, 0], dmaTasks = [[], []], invariantCount = [],
          managedMappedInference = @program.mapped_inference::@MappedInference_managed,
          mappedInferenceVersion = @note.MappedInferenceVersion::@MappedInferenceVersion_0_0, mediaCount = 0 : i64,
          sym_name = "MappedInference", variantCount = []}> : () -> ()
        "NPUReg40XX.ManagedMappedInference"() <{barrierTasks = @program.managedBarrier::@ConfigureBarrier_0_0, bootstrapTasks = @program.bootstrap::@Bootstrap_0_0, dmaTasks = [[], []], descriptor = #NPUReg40XX.VpuManagedMappedInference<
          VpuManagedMappedInference {
            MMI_vpu_nnrt_api_ver = UINT 0,
            MMI_final_barrier = UINT 0,
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
            MMI_pad1_0 = UINT 0,
            MMI_pad1_1 = UINT 0,
            MMI_pad1_2 = UINT 0,
            MMI_pad1_3 = UINT 0,
            MMI_pad1_4 = UINT 0,
            MMI_pad1_5 = UINT 0,
            MMI_pad1_6 = UINT 0,
            MMI_pad1_7 = UINT 0,
            MMI_pad1_8 = UINT 0,
            MMI_pad1_9 = UINT 0,
            MMI_pad1_10 = UINT 0,
            MMI_pad1_11 = UINT 0,
            MMI_pad1_12 = UINT 0,
            MMI_pad1_13 = UINT 0,
            MMI_pad1_14 = UINT 0,
            MMI_pad1_15 = UINT 0,
            MMI_pad1_16 = UINT 0,
            MMI_pad1_17 = UINT 0,
            MMI_pad1_18 = UINT 0,
            MMI_pad1_19 = UINT 0,
            MMI_pad1_20 = UINT 0,
            MMI_pad1_21 = UINT 0,
            MMI_pad1_22 = UINT 0,
            MMI_pad1_23 = UINT 0,
            MMI_pad1_24 = UINT 0,
            MMI_pad1_25 = UINT 0,
            MMI_pad1_26 = UINT 0,
            MMI_pad1_27 = UINT 0,
            MMI_pad1_28 = UINT 0,
            MMI_pad1_29 = UINT 0,
            MMI_bootstrap_workitems_count = UINT 0,
            MMI_pad2_ = UINT 0
          } requires 11:5:0
        >, mappedInferenceVersion = @note.MappedInferenceVersion::@MappedInferenceVersion_0_0, nnrtConfig = @program.nnrt_config::@MappedInference_nnrtConfigManaged, sym_name = "MappedInference_managed", workItems = @program.workItem::@ed.Enqueue_0_0}> : () -> ()
      }

    }
    return
  }

}
//CHECK: ELF.CreateSection @note.MappedInferenceVersion
//CHECK:       NPUReg40XX.MappedInferenceVersion(11 _ 5 _ 0)
//CHECK: ELF.CreateSection @program.mapped_inference
//CHECK: "NPUReg40XX.ManagedMappedInference"
//CHECK: VpuManagedMappedInference
//CHECK:             MMI_vpu_nnrt_api_ver = UINT 0xB0005
//CHECK: requires 11:5:0

// -----

module @setDynMi {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main() {
    ELF.Main @ELFMain {
      ELF.CreateSection @program.workItem aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") {
        "NPUReg40XX.WorkItem"() <{first_task = @task.dma.0.0::@NNDMA_0_0_0, sym_name = "ed.Enqueue_0_0", task_type = #VPURegMapped.task_type<DMA>, descriptor = #NPUReg40XX.WorkItem<
          WorkItem {
            desc_ptr = UINT 0,
            wi_type = UINT 1,
            wi_unit = UINT 0,
            wi_sub_unit = UINT 0,
            pad0_8_0 = UINT 0,
            next_workitem_idx = UINT 87,
            pad1_8_0 = UINT 0,
            pad1_8_1 = UINT 0,
            pad1_8_2 = UINT 0,
            pad1_8_3 = UINT 0,
            pad1_8_4 = UINT 0,
            pad1_8_5 = UINT 0
          } requires 11:6:0
        >}> : () -> ()
      }
      ELF.CreateSection @note.MappedInferenceVersion aligned(64) secType(SHT_NOTE) secFlags("SHF_NONE") {
        NPUReg40XX.MappedInferenceVersion(11 _ 4 _ 10) {sym_name = "MappedInferenceVersion_0_0"}
      }
      ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") {
        "NPUReg40XX.MappedInference"() <{actKernelInvocations = [], actKernelInvocationsCount = [], actKernelRanges = [], actKernelRangesCount = [], barrierCount = 0 : i64, dmaCMXCount = [0, 0], dmaCount = [[0, 0], [0, 0]], dmaDDRCount = [0, 0], dmaTasks = [[], []], invariantCount = [],
          managedMappedInference = @program.mapped_inference::@MappedInference_managed,
          mappedInferenceVersion = @note.MappedInferenceVersion::@MappedInferenceVersion_0_0, mediaCount = 0 : i64,
          sym_name = "MappedInference", variantCount = []}> : () -> ()
        "NPUReg40XX.ManagedMappedInference"() <{barrierTasks = @program.managedBarrier::@ConfigureBarrier_0_0, bootstrapTasks = @program.bootstrap::@Bootstrap_0_0, dmaTasks = [[], []], descriptor = #NPUReg40XX.VpuManagedMappedInference<
          VpuManagedMappedInference {
            MMI_vpu_nnrt_api_ver = UINT 0,
            MMI_final_barrier = UINT 0,
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
            MMI_pad1_0 = UINT 0,
            MMI_pad1_1 = UINT 0,
            MMI_pad1_2 = UINT 0,
            MMI_pad1_3 = UINT 0,
            MMI_pad1_4 = UINT 0,
            MMI_pad1_5 = UINT 0,
            MMI_pad1_6 = UINT 0,
            MMI_pad1_7 = UINT 0,
            MMI_pad1_8 = UINT 0,
            MMI_pad1_9 = UINT 0,
            MMI_pad1_10 = UINT 0,
            MMI_pad1_11 = UINT 0,
            MMI_pad1_12 = UINT 0,
            MMI_pad1_13 = UINT 0,
            MMI_pad1_14 = UINT 0,
            MMI_pad1_15 = UINT 0,
            MMI_pad1_16 = UINT 0,
            MMI_pad1_17 = UINT 0,
            MMI_pad1_18 = UINT 0,
            MMI_pad1_19 = UINT 0,
            MMI_pad1_20 = UINT 0,
            MMI_pad1_21 = UINT 0,
            MMI_pad1_22 = UINT 0,
            MMI_pad1_23 = UINT 0,
            MMI_pad1_24 = UINT 0,
            MMI_pad1_25 = UINT 0,
            MMI_pad1_26 = UINT 0,
            MMI_pad1_27 = UINT 0,
            MMI_pad1_28 = UINT 0,
            MMI_pad1_29 = UINT 0,
            MMI_bootstrap_workitems_count = UINT 0,
            MMI_pad2_ = UINT 0
          } requires 11:5:0
        >, mappedInferenceVersion = @note.MappedInferenceVersion::@MappedInferenceVersion_0_0, nnrtConfig = @program.nnrt_config::@MappedInference_nnrtConfigManaged, sym_name = "MappedInference_managed", workItems = @program.workItem::@ed.Enqueue_0_0}> : () -> ()
      }

    }
    return
  }
}

//CHECK:  ELF.CreateSection @program.workItem
//CHECK:  "NPUReg40XX.WorkItem"()
//CHECK: requires 11:6:0
//CHECK: ELF.CreateSection @note.MappedInferenceVersion
//CHECK:       NPUReg40XX.MappedInferenceVersion(11 _ 6 _ 0)
//CHECK: ELF.CreateSection @program.mapped_inference
//CHECK: "NPUReg40XX.ManagedMappedInference"

// mapped inference version required 11.5, but the max version used in the IR
// comes from workItem which has 11.6

//CHECK: VpuManagedMappedInference
//CHECK:             MMI_vpu_nnrt_api_ver = UINT 0xB0006
//CHECK: requires 11:5:0
