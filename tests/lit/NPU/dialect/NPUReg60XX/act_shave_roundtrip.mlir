//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: dev-build && platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @ActShaveTest{
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x10x2x3xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x10x2x3xf16>
  }
  func.func @main() {
    ELF.Main {
      VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x10x2x3xf16, {order = #NHWC}, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x10x2x3xf16, {order = #NHWC}, @DDR> :  swizzling(0)>
      VPUASM.DeclareKernelEntry @DeclareKernelEntry0 : "activation_sigmoid"
      ELF.CreateLogicalSection @program.ActKernelRange.cmx.0.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_ActKernelRange_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <ActKernelRange>
      }
      ELF.CreateLogicalSection @program.ActKernelInvocation.cmx.0.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_ActKernelInvocation_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <ActKernelInvocation>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer2 !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x10x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer5 !VPUASM.Buffer< "CMX_NN"[0] <120> : memref<1x10x2x3xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateSection @text.shave aligned(1024) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_SHAVE") secLocation(<DDR>) {
        VPUASM.DeclareKernelText @DeclareKernelText0 : "activation_sigmoid"
      }
      ELF.CreateSection @program.shave.data aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_SHAVE") secLocation(<DDR>) {
        VPUASM.DeclareKernelData @DeclareKernelArgs0 : "activation_sigmoid"
      }
      ELF.CreateSection @program.shave.parameter aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_SHAVE") secLocation(<DDR>) {
        VPUASM.KernelParams @KernelParams0 inputs([@buffer.CMX_NN.0::@DeclareBuffer2]) outputs([@buffer.CMX_NN.0::@DeclareBuffer5]) dynamicInputShapes([]) dynamicOutputShapes([]) kernel_type("activation_sigmoid") < {
          inputDimsBinaryVector = [], inputStridesBinaryVector = [],
          kernel_params = [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 49, 36, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 49, 36, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
          outputDimsBinaryVector = [], outputStridesBinaryVector = []}>
      }
      ELF.CreateSection @task.shave.range.0.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") secLocation(<DDR>) {
        NPUReg60XX.ActKernelRange descriptor = <
          VpuActKernelRange {
            type = UINT 0,
            kernel_entry = UINT 0x1D000000,
            text_window_base = UINT 0,
            code_size = UINT 0x620,
            deprecated_akr = UINT 0,
            kernel_invo_count = UINT 0,
            pad1_4 = UINT 0,
          } requires 11:13:2
        > {kernel_entry = @DeclareKernelEntry0, kernel_text = @text.shave::@DeclareKernelText0, sym_name = "ActKernelRange0", task_location = @program.ActKernelRange.cmx.0.0::@DeclareTaskBuffer_ActKernelRange_0_0_0}
        // CHECK: NPUReg60XX.ActKernelRange descriptor = <
        // CHECK:   VpuActKernelRange {
        // CHECK:     type = UINT 0,
        // CHECK:     kernel_entry = UINT 0x1D000000,
        // CHECK:     text_window_base = UINT 0,
        // CHECK:     code_size = UINT 0x620,
        // CHECK:     deprecated_akr = UINT 0,
        // CHECK:     kernel_invo_count = UINT 0,
        // CHECK:     pad1_4 = UINT 0,
        // CHECK:   } requires 11:13:3
        // CHECK: >
      }
      ELF.CreateSection @task.shave.invocation.0.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") secLocation(<DDR>) {
        NPUReg60XX.ActKernelInvocation descriptor = <
          VpuActKernelInvocation {
            range = UINT 0x200000,
            kernel_args = UINT 0,
            data_window_base = UINT 0,
            perf_packet_out = UINT 0,
            barriers_wait_mask_hi_act {
              UINT barriers_wait_mask_hi_act = 0,
            }
            barriers_wait_mask_lo_act = UINT 1,
            barriers_post_mask_hi_act {
              UINT barriers_post_mask_hi_act = 0,
            }
            barriers_post_mask_lo_act = UINT 2,
            barriers_group_mask_act {
              UINT group_act = 1,
              UINT mask_act = 1,
            }
            act_invo_barriers_sched {
              UINT start_after_ = 2,
              UINT clean_after_ = 1,
            }
            invo_index = UINT 0,
            invo_tile = UINT 0,
            kernel_range_index = UINT 0,
            next_aki_wl_addr = UINT 0,
          } requires 11:13:2
        > {kernel_data = @program.shave.data::@DeclareKernelArgs0, kernel_params = @program.shave.parameter::@KernelParams0,
         kernel_range = @program.ActKernelRange.cmx.0.0::@DeclareTaskBuffer_ActKernelRange_0_0_0, sym_name = "ActKernelInvocation0", task_location = @program.ActKernelInvocation.cmx.0.0::@DeclareTaskBuffer_ActKernelInvocation_0_0_0}
        // CHECK: NPUReg60XX.ActKernelInvocation descriptor = <
        // CHECK:   VpuActKernelInvocation {
        // CHECK:     range = UINT 0x200000,
        // CHECK:     kernel_args = UINT 0,
        // CHECK:     data_window_base = UINT 0,
        // CHECK:     perf_packet_out = UINT 0,
        // CHECK:     barriers_wait_mask_hi_act {
        // CHECK:       UINT barriers_wait_mask_hi_act = 0,
        // CHECK:     }
        // CHECK:     barriers_wait_mask_lo_act = UINT 1,
        // CHECK:     barriers_post_mask_hi_act {
        // CHECK:       UINT barriers_post_mask_hi_act = 0,
        // CHECK:     }
        // CHECK:     barriers_post_mask_lo_act = UINT 2,
        // CHECK:     barriers_group_mask_act {
        // CHECK:       UINT group_act = 1,
        // CHECK:       UINT mask_act = 1,
        // CHECK:     }
        // CHECK:     act_invo_barriers_sched {
        // CHECK:       UINT start_after_ = 2,
        // CHECK:       UINT clean_after_ = 1,
        // CHECK:     }
        // CHECK:     invo_index = UINT 0,
        // CHECK:     invo_tile = UINT 0,
        // CHECK:     kernel_range_index = UINT 0,
        // CHECK:     next_aki_wl_addr = UINT 0,
        // CHECK:   } requires 11:13:3
        // CHECK: >
      }
      ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
        ELF.Symbol @elfsym.program.ActKernelRange.cmx.0.0 of(@program.ActKernelRange.cmx.0.0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.program.ActKernelInvocation.cmx.0.0 of(@program.ActKernelInvocation.cmx.0.0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.text.shave of(@text.shave) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.program.shave.data of(@program.shave.data) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.program.shave.parameter of(@program.shave.parameter) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.task.shave.range.0.0 of(@task.shave.range.0.0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.task.shave.invocation.0.0 of(@task.shave.invocation.0.0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @entry of(@program.mapped_inference::@MappedInference) type(<VPU_STT_ENTRY>) size(0) value(0)
      }
      ELF.CreateMetadataSection @MetadataSection aligned(8) secFlags("SHF_NONE")  {
        VPUASM.NetworkMetadata @NetworkMetadata
      }
    }
    return
  }
}
