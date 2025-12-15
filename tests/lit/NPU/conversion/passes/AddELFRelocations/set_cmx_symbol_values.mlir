//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --set-cmx-symbol="cmx-workspace-addr=1075937280 cmx-workspace-size=1474560 cmx-metadata-addr=1075854336 cmx-metadata-size=82944" %s | FileCheck %s
// REQUIRES: dev-build && (arch-NPU40XX || arch-NPU50XX)

module @setCMXSymbols {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main() attributes {inliner_dispatch = #VPUIP.VPUIPInlinerDispatch} {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @io.NetworkInput.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USERINPUT|VPU_SHF_PROC_DMA") secLocation(<NetworkInput>) {
      }
      ELF.CreateLogicalSection @io.NetworkOutput.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USEROUTPUT|VPU_SHF_PROC_DMA") secLocation(<NetworkOutput>) {
      }
      ELF.CreateLogicalSection @program.metadata.cmx aligned(1) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
      }
      ELF.CreateLogicalSection @buffer.DDR.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_PROC_DMA") secLocation(<DDR>) {
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.1 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.2 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.3 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
      }
      ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") secLocation(<DDR>) {
        "NPUReg40XX.MappedInference"() <{actKernelInvocations = [@task.shave.invocation.0.0::@ActKernelInvocation_0_0, @task.shave.invocation.1.0::@ActKernelInvocation_1_0, @task.shave.invocation.2.0::@ActKernelInvocation_2_0, @task.shave.invocation.3.0::@ActKernelInvocation_3_0], actKernelInvocationsCount = [3, 3, 3, 3], actKernelRanges = [@task.shave.range.0.0::@ActKernelRange_0_0, @task.shave.range.1.0::@ActKernelRange_1_0, @task.shave.range.2.0::@ActKernelRange_2_0, @task.shave.range.3.0::@ActKernelRange_3_0], actKernelRangesCount = [3, 3, 3, 3], actShaveRt = @shave.runtime::@ActShaveRt, barrierCount = 5 : i64, barrierTasks = @program.managedBarrier::@ConfigureBarrier_0_0, dmaCMXCount = [2, 2], dmaCount = [[11, 2], [2, 2]], dmaDDRCount = [11, 2], dmaHwpBase = @buffer.CMX_NN.0::@DeclareBuffer12, dmaTasks = [[@task.dma.0.0::@NNDMA_0_0_0, @task.dma.0.1::@NNDMA_0_1_0], [@task.dma.1.0::@NNDMA_1_0_0, @task.dma.1.1::@NNDMA_1_1_0]], invariantCount = [0, 0, 0, 0], managedMappedInference = @program.mapped_inference::@MappedInference_managed, mappedInferenceVersion = @note.MappedInferenceVersion::@MappedInferenceVersion_0_0, mediaCount = 0 : i64, sym_name = "MappedInference", variantCount = [0, 0, 0, 0]}> {elfMemOffsetAttrKey = 0 : ui64} : () -> ()
        NPUReg40XX.ManagedMappedInference descriptor = <
          VpuManagedMappedInference {
            MMI_vpu_nnrt_api_ver = UINT 0xB0004,
            MMI_final_barrier = UINT 4,
            taskReferenceR1_MMI_work_item = UINT 0,
            taskReferenceR2_MMI_work_item = UINT 0,
            taskReferenceR3_MMI_work_item = UINT 0,
            taskReferenceAddr_MMI_work_item = UINT 0,
            taskReferenceCount_MMI_work_item = UINT 0xC,
            taskReferenceR1_MMI_task_configs = UINT 0,
            taskReferenceR2_MMI_task_configs = UINT 0,
            taskReferenceR3_MMI_task_configs = UINT 0,
            taskReferenceAddr_MMI_task_configs = UINT 0,
            taskReferenceCount_MMI_task_configs = UINT 5,
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
            taskReferenceCount_MMI_barriers_configuration = UINT 0,
            taskReferenceR1_MMI_num_of_barrier_reprogrammings = UINT 0,
            taskReferenceR2_MMI_num_of_barrier_reprogrammings = UINT 0,
            taskReferenceR3_MMI_num_of_barrier_reprogrammings = UINT 0,
            taskReferenceAddr_MMI_num_of_barrier_reprogrammings = UINT 0,
            taskReferenceCount_MMI_num_of_barrier_reprogrammings = UINT 0,
            taskReferenceR1_MMI_initial_barriers = UINT 0,
            taskReferenceR2_MMI_initial_barriers = UINT 0,
            taskReferenceR3_MMI_initial_barriers = UINT 0,
            taskReferenceAddr_MMI_initial_barriers = UINT 0,
            taskReferenceCount_MMI_initial_barriers = UINT 5,
            taskReferenceR1_MMI_nnrt_config = UINT 0,
            taskReferenceR2_MMI_nnrt_config = UINT 0,
            taskReferenceR3_MMI_nnrt_config = UINT 0,
            taskReferenceAddr_MMI_nnrt_config = UINT 0,
            taskReferenceCount_MMI_nnrt_config = UINT 1,
            MMI_actshv_used = UINT 0xF,
            MMI_dpu_used = UINT 0xF,
            MMI_media_used = UINT 0,
            MMI_dma_from_ddr_used = UINT 3,
            MMI_dma_from_cmx_used = UINT 3,
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
            MMI_model_identifier = UINT 0,
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
            MMI_bootstrap_workitems_count = UINT 4,
            MMI_pad3_ = UINT 0,
          } requires 11:4:10
        > {barrierTasks = @program.managedBarrier::@ConfigureBarrier_0_0,
        bootstrapTasks = @program.bootstrap::@Bootstrap_0_0, dmaTasks = [[], []], elfMemOffsetAttrKey = 1728 : ui64,
         mappedInferenceVersion = @note.MappedInferenceVersion::@MappedInferenceVersion_0_0,
         nnrtConfig = @program.nnrt_config::@MappedInference_nnrtConfigManaged, sym_name = "MappedInference_managed",
         workItems = @program.workItem::@ed.Enqueue_0_0}
      }
      ELF.CreateSymbolTableSection @symtab.io.NetworkInput secFlags("VPU_SHF_JIT|VPU_SHF_USERINPUT") {
        ELF.Symbol @elfsym.io.NetworkInput.0 of(@io.NetworkInput.0) type(<STT_SECTION>) size(256) value(0)
      }
      ELF.CreateSymbolTableSection @symtab.io.NetworkOutput secFlags("VPU_SHF_JIT|VPU_SHF_USEROUTPUT") {
        ELF.Symbol @elfsym.io.NetworkOutput.0 of(@io.NetworkOutput.0) type(<STT_SECTION>) size(256) value(0)
      }
      ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
        ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.buffer.DDR.0 of(@buffer.DDR.0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.buffer.CMX_NN.1 of(@buffer.CMX_NN.1) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.buffer.CMX_NN.2 of(@buffer.CMX_NN.2) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.buffer.CMX_NN.3 of(@buffer.CMX_NN.3) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @entry of(@program.mapped_inference::@MappedInference) type(<VPU_STT_ENTRY>) size(0) value(0)
      }
    }
    return
  }
}


//CHECK: ELF.CreateSymbolTableSection @symtab.io.NetworkInput secFlags("VPU_SHF_JIT|VPU_SHF_USERINPUT")
//CHECK:   ELF.Symbol @elfsym.io.NetworkInput.0 of(@io.NetworkInput.0) type(<STT_SECTION>) size(256) value(0)

//CHECK: ELF.CreateSymbolTableSection @symtab.io.NetworkOutput secFlags("VPU_SHF_JIT|VPU_SHF_USEROUTPUT")
//CHECK:   ELF.Symbol @elfsym.io.NetworkOutput.0 of(@io.NetworkOutput.0) type(<STT_SECTION>) size(256) value(0)

//CHECK: ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE")
//CHECK:   ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>) size(82944) value(1075854336)
//CHECK:   ELF.Symbol @elfsym.buffer.DDR.0 of(@buffer.DDR.0) type(<STT_SECTION>) size(0) value(0)
//CHECK:   ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>) size(1474560) value(1075937280)
//CHECK:   ELF.Symbol @elfsym.buffer.CMX_NN.1 of(@buffer.CMX_NN.1) type(<STT_SECTION>) size(1474560) value(1075937280)
//CHECK:   ELF.Symbol @elfsym.buffer.CMX_NN.2 of(@buffer.CMX_NN.2) type(<STT_SECTION>) size(1474560) value(1075937280)
//CHECK:   ELF.Symbol @elfsym.buffer.CMX_NN.3 of(@buffer.CMX_NN.3) type(<STT_SECTION>) size(1474560) value(1075937280)
//CHECK:   ELF.Symbol @entry of(@program.mapped_inference::@MappedInference) type(<VPU_STT_ENTRY>) size(0) value(0)
