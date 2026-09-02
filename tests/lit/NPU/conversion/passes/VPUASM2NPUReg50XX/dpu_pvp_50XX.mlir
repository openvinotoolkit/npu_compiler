//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This test verifies the NPU50XX lowering of the DpuPVP task:
//   - the VPUASM.DpuPVP op is erased (DpuPVP is not needed on NPU50XX);
//   - the resulting NPUReg50XX.ManagedMappedInference carries no dpuPvpTask reference;
//   - the now-empty @dpu.pvp section is dropped by the empty-section cleanup pass.

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --cmx-stack-frames-reserve-mem --convert-VPUASM-to-NPUReg50XX --remove-empty-ELF-sections %s | FileCheck %s
// REQUIRES: dev-build && platform-NPU5010

module @DpuPvpErasedOnNpu50XX {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  VPUASM.InputBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
  }
  VPUASM.OutputBindings outputDeclarations : {
    VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
  }
  VPUASM.ProfilingBindings profilingDeclarations : {
  }
  func.func @main() {
    ELF.Main {
      VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>

      ELF.CreateLogicalSection @builtin.tasks.DMA0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0 idx(!VPURegMapped.Index<0:0:0>) <DMA>
      }

      ELF.CreateSection @text.nndma0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@builtin.tasks.DMA0::@DeclareTaskBuffer_DMA_0) input(@DeclareBuffer0) outputs([@DeclareBuffer1]) waits([]) updates([]) start_after(1) clean_after(2) dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 48 : i32, srcWidth = 48 : i32, srcStride = 48 : i32, srcPlaneStride = 48 : i32, dstWidth = 48 : i32, dstStride = 48 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>)
      }

      ELF.CreateSection @note.MappedInferenceVersion aligned(4) secType(SHT_NOTE) secFlags("SHF_NONE") secLocation(<DDR>) {
        VPUASM.MappedInferenceVersion @MappedInferenceVersion_0_0(11 _ 4 _ 10)
      }

      ELF.CreateSection @program.nnrt_config aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") secLocation(<DDR>) {
        VPUASM.nnrtConfig @MappedInference_nnrtConfigManaged :
      }

      ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DpuPVP @DpuPVP_0_0_0
      }

      ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC") secLocation(<DDR>) {
        VPUASM.ManagedMappedInference @MappedInference_managed : dmas ([[@text.nndma0::@NNDMA_0_0_0]]) workItems() barrierTasks() bootstrapBarriers() nnrtConfig(@program.nnrt_config::@MappedInference_nnrtConfigManaged) mappedInferenceVersion(@note.MappedInferenceVersion::@MappedInferenceVersion_0_0) dpuPvpTask(@dpu.pvp::@DpuPVP_0_0_0) {actshv_used = 0 : ui8, barrierConfigurationStride = 0 : i64, barrierConfigurationTasksCount = 0 : i64, barrierCount = 0 : i64, barriersReprogrammingCount = 0 : i64, bootstrapWorkItemsCount = 0 : i64, bootstrapBarriersCount = 0 : i64, disableDmaSwFifo = true, dmaCount = [[0, 0], [0, 0]], dma_from_cmx_used = 0 : ui8, dma_from_ddr_used = 1 : ui8, dpu_used = 0 : ui8, final_barrier_id = 0 : i64, media_used = 0 : ui8, workItemsCount = 1 : i64, workloadManagementBarrierProgrammingMode = #VPURegMapped.workload_management_barrier_programming_mode<ALL_BARRIER_DMAS_SCHEDULED>}
      }
    }
    return
  }
}

// The DpuPVP op is erased and its section is dropped by the cleanup pass.
// CHECK-NOT: VPUASM.DpuPVP
// CHECK-NOT: @dpu.pvp

// The lowered ManagedMappedInference does not reference any DpuPVP task.
// CHECK: NPUReg50XX.ManagedMappedInference
// CHECK-NOT: dpuPvpTask
