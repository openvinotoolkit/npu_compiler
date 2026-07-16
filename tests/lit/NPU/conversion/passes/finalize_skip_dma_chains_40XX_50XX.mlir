//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --platform=%platform% --finalize-skip-dma-chains %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @Activation attributes {config.compilationMode = #config.compilation_mode<DefaultHW>, config.elf_version = #config.version<2 : 0 : 0>, config.platform = #config.platform<NPU5010>, config.revisionID = #config.revision_id<REVISION_NONE>} {
  config.Resources {activity_factor = 0.000000e+00 : f64} 1 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.Resources 1 of @global {
    config.ExecutorResource 1 of @M2I
    config.ExecutorResource 2 of @DMA_NN
    config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  }
  net.NetworkInfo {inferenceTiming = 237047 : i64} entryPoint : @main inputsInfo : {
    DataInfo "Input" tensorNames = ["dyn_input"] : tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 4194304]> : tensor<4xsi64>, order = #NCHW}>
    DataInfo "vpux_ie_shape_Input" : tensor<4xsi32>
  } outputsInfo : {
    DataInfo "Sigm_11" friendlyName = "Result_12" : tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 4194304]> : tensor<4xsi64>, order = #NCHW}>
    DataInfo "vpux_ie_shape_Sigm_11" : tensor<4xsi32>
  }
  func.func @main() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DMA> {offset = 55552 : ui64}
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_0_1 idx(!VPURegMapped.Index<0:0:1>) <DMA> {offset = 55776 : ui64}
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_1_0 idx(!VPURegMapped.Index<0:1:0>) <DMA> {offset = 56000 : ui64}
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_1_1 idx(!VPURegMapped.Index<0:1:1>) <DMA> {offset = 56224 : ui64}
      }

      ELF.CreateSection @shave.params aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.KernelParams @KernelParams_0_0_0 inputs([@buffer.DDR.0::@DeclareBuffer_26, @buffer.CMX_NN.0::@DeclareBuffer_28]) outputs([@buffer.DDR.0::@DeclareBuffer_21, @buffer.CMX_NN.0::@DeclareBuffer_28]) dynamicInputShapes([@buffer.DDR.0::@DeclareBuffer_23, @placeholder_symbol]) dynamicOutputShapes([@buffer.DDR.0::@DeclareBuffer_22, @placeholder_symbol]) kernel_type("activation_atan_dma") <{inputDimsBinaryVector = [0, 0, 64, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 4, 16, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], inputStridesBinaryVector = [16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 32, 128, 0, 0, 0, 0, 0, 0, 32, 128, 0, 0, 0, 0, 0, 0, 32, 128, 0, 0, 0, 0, 0], kernel_params = [0, 0, 32, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 32, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 32, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0], outputDimsBinaryVector = [0, 0, 64, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 4, 16, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], outputStridesBinaryVector = [16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 32, 128, 0, 0, 0, 0, 0, 0, 32, 128, 0, 0, 0, 0, 0, 0, 32, 128, 0, 0, 0, 0, 0], skipDescIds = [0, 1]}>
        VPUASM.KernelParams @KernelParams_0_1_0 inputs([@buffer.DDR.0::@DeclareBuffer_26, @buffer.CMX_NN.0::@DeclareBuffer_28]) outputs([@buffer.DDR.0::@DeclareBuffer_21, @buffer.CMX_NN.0::@DeclareBuffer_28]) dynamicInputShapes([@buffer.DDR.0::@DeclareBuffer_23, @placeholder_symbol]) dynamicOutputShapes([@buffer.DDR.0::@DeclareBuffer_22, @placeholder_symbol]) kernel_type("activation_atan_dma") <{inputDimsBinaryVector = [0, 0, 64, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 4, 16, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], inputStridesBinaryVector = [16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 32, 128, 0, 0, 0, 0, 0, 0, 32, 128, 0, 0, 0, 0, 0, 0, 32, 128, 0, 0, 0, 0, 0], kernel_params = [0, 0, 32, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 32, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 32, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0], outputDimsBinaryVector = [0, 0, 64, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 4, 16, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], outputStridesBinaryVector = [16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 32, 128, 0, 0, 0, 0, 0, 0, 32, 128, 0, 0, 0, 0, 0, 0, 32, 128, 0, 0, 0, 0, 0], skipDescIds = [2, 3]}>
      }

      ELF.CreateSection @task.dma.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        // Fetch DMAs for Skip DMAs
        VPUASM.NNDMA @NNDMA_0_0_10 idx(!VPURegMapped.Index<0:0:10>) links(@task.dma.0.0::@NNDMA_0_0_11) input(@buffer.DDR.0::@DeclareBuffer_10) outputs([@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_0_0]) waits([]) updates([]) start_after(0) clean_after(0) dma_descriptor(<numPlanes = 0 : i32, len = 224 : i32, srcWidth = 224 : i32, srcStride = 224 : i32, srcPlaneStride = 0 : i32, dstWidth = 224 : i32, dstStride = 224 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>) is_out_of_order() is_critical() tile_indexes([0]) <{fetch_dma = #VPUIP.FetchDMAAttr<<DMA_NN>, tile = 0 : i64, list = 0 : i64, fetchType = <SingleDescriptor>, logicalTaskIdx = 0 : i64, descId = 0 : i64>}>
        VPUASM.NNDMA @NNDMA_0_0_11 idx(!VPURegMapped.Index<0:0:11>) links(@task.dma.0.0::@NNDMA_0_0_12) input(@buffer.DDR.0::@DeclareBuffer_10) outputs([@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_0_1]) waits([]) updates([]) start_after(0) clean_after(0) dma_descriptor(<numPlanes = 0 : i32, len = 224 : i32, srcWidth = 224 : i32, srcStride = 224 : i32, srcPlaneStride = 0 : i32, dstWidth = 224 : i32, dstStride = 224 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>) is_out_of_order() is_critical() tile_indexes([0]) <{fetch_dma = #VPUIP.FetchDMAAttr<<DMA_NN>, tile = 0 : i64, list = 0 : i64, fetchType = <SingleDescriptor>, logicalTaskIdx = 0 : i64, descId = 1 : i64>}>
        VPUASM.NNDMA @NNDMA_0_0_12 idx(!VPURegMapped.Index<0:0:12>) links(@task.dma.0.0::@NNDMA_0_0_13) input(@buffer.DDR.0::@DeclareBuffer_10) outputs([@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_1_0]) waits([]) updates([]) start_after(0) clean_after(0) dma_descriptor(<numPlanes = 0 : i32, len = 224 : i32, srcWidth = 224 : i32, srcStride = 224 : i32, srcPlaneStride = 0 : i32, dstWidth = 224 : i32, dstStride = 224 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>) is_out_of_order() is_critical() tile_indexes([0]) <{fetch_dma = #VPUIP.FetchDMAAttr<<DMA_NN>, tile = 0 : i64, list = 1 : i64, fetchType = <SingleDescriptor>, logicalTaskIdx = 0 : i64, descId = 2 : i64>}>
        VPUASM.NNDMA @NNDMA_0_0_13 idx(!VPURegMapped.Index<0:0:13>) links(@task.dma.0.0::@NNDMA_0_0_14) input(@buffer.DDR.0::@DeclareBuffer_10) outputs([@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_1_1]) waits([]) updates([1 : ui16, 2 : ui16]) start_after(0) clean_after(0) dma_descriptor(<numPlanes = 0 : i32, len = 224 : i32, srcWidth = 224 : i32, srcStride = 224 : i32, srcPlaneStride = 0 : i32, dstWidth = 224 : i32, dstStride = 224 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>) is_out_of_order() is_critical() tile_indexes([0]) <{fetch_dma = #VPUIP.FetchDMAAttr<<DMA_NN>, tile = 0 : i64, list = 1 : i64, fetchType = <SingleDescriptor>, logicalTaskIdx = 0 : i64, descId = 3 : i64>}>

        // Sync DMA
        VPUASM.NNDMA @NNDMA_0_0_14 idx(!VPURegMapped.Index<0:0:14>) links(@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_0_0) input(@buffer.DDR.0::@DeclareBuffer_18) outputs([@buffer.DDR.0::@DeclareBuffer_17]) waits([1 : ui16]) updates([]) start_after(0) clean_after(0) dma_descriptor(<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>) acceleration_mode(<DISABLE>) is_out_of_order()

        // Skip DMAs for DDR channel
        VPUASM.NNDMA @NNDMA_0_0_15 idx(!VPURegMapped.Index<0:0:15>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_0_0) links(@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_1_0) input(@buffer.DDR.0::@DeclareBuffer_18) outputs([@buffer.DDR.0::@DeclareBuffer_17]) waits([]) updates([]) start_after(0) clean_after(0) dma_descriptor(<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>) acceleration_mode(<DISABLE>) is_out_of_order() <{skip_dma = #VPUIP.SkipDMAAttr<tile = 0 : i64, list = 0 : i64, logicalTask = 0 : i64, descId = 0 : i64>}>
        VPUASM.NNDMA @NNDMA_0_0_16 idx(!VPURegMapped.Index<0:0:16>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_1_0) links(@task.dma.0.0::@NNDMA_0_0_17) input(@buffer.DDR.0::@DeclareBuffer_18) outputs([@buffer.DDR.0::@DeclareBuffer_17]) waits([]) updates([]) start_after(0) clean_after(0) dma_descriptor(<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>) acceleration_mode(<DISABLE>) is_out_of_order() <{skip_dma = #VPUIP.SkipDMAAttr<tile = 0 : i64, list = 1 : i64, logicalTask = 0 : i64, descId = 2 : i64>}>

        // Release DMA for DDR Channel
        VPUASM.NNDMA @NNDMA_0_0_17 idx(!VPURegMapped.Index<0:0:17>) links(@task.dma.0.0::@NNDMA_0_0_18) input(@buffer.DDR.0::@DeclareBuffer_22) outputs([@io.NetworkOutput.1::@DeclareBuffer_9]) waits([3 : ui16]) updates([]) start_after(0) clean_after(0) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<4xsi32, @DDR>, outputType = memref<4xsi32, @DDR>>) acceleration_mode(<DISABLE>) is_out_of_order()
        VPUASM.NNDMA @NNDMA_0_0_18 idx(!VPURegMapped.Index<0:0:18>) input(@buffer.DDR.0::@DeclareBuffer_20) outputs([@io.NetworkOutput.0::@DeclareBuffer_8]) waits([]) updates([4 : ui16]) start_after(0) clean_after(0) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x1x1x2097152xf16,  @DDR>, outputType = memref<1x1x1x2097152xf16,  @DDR>>) acceleration_mode(<DISABLE>) is_out_of_order()
      }

      ELF.CreateSection @task.dma.0.1 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        // Skip DMA for CMX channel
        VPUASM.NNDMA @NNDMA_0_1_1 idx(!VPURegMapped.Index<0:1:1>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_0_1) links(@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_1_1) input(@buffer.CMX_NN.0::@DeclareBuffer_27) outputs([@buffer.DDR.0::@DeclareBuffer_17]) waits([]) updates([]) start_after(0) clean_after(0) dma_descriptor(<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>) acceleration_mode(<DISABLE>) is_out_of_order() <{enable_msc, skip_dma = #VPUIP.SkipDMAAttr<tile = 0 : i64, list = 0 : i64, logicalTask = 0 : i64, descId = 1 : i64>}>
        VPUASM.NNDMA @NNDMA_0_1_2 idx(!VPURegMapped.Index<0:1:2>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_1_1) links(@task.dma.0.1::@NNDMA_0_1_3) input(@buffer.CMX_NN.0::@DeclareBuffer_27) outputs([@buffer.DDR.0::@DeclareBuffer_17]) waits([]) updates([]) start_after(0) clean_after(0) dma_descriptor(<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>) acceleration_mode(<DISABLE>) is_out_of_order() <{enable_msc, skip_dma = #VPUIP.SkipDMAAttr<tile = 0 : i64, list = 1 : i64, logicalTask = 0 : i64, descId = 3 : i64>}>

        // Release DMA for CMX Channel
        VPUASM.NNDMA @NNDMA_0_1_3 idx(!VPURegMapped.Index<0:1:3>) input(@buffer.CMX_NN.0::@DeclareBuffer_27) outputs([@buffer.DDR.0::@DeclareBuffer_17]) waits([]) updates([4 : ui16]) start_after(0) clean_after(0) dma_descriptor(<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>) acceleration_mode(<DISABLE>) is_out_of_order() <{enable_msc}>
      }

      ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.MappedInference @MappedInference : dmas([[@task.dma.0.0::@NNDMA_0_0_0, @task.dma.0.1::@NNDMA_0_1_0], [@task.dma.1.0::@NNDMA_1_0_0]]) actKernelRanges([@task.shave.range.0.0::@ActKernelRange_0_0_0]) actKernelInvocations([@task.shave.invocation.0.0::@ActKernelInvocation_0_0_0]) barriers(@program.managedBarrier::@ConfigureBarrier_0_0_0) actShaveRt(@shave.runtime::@ActShaveRt) managedMappedInference(@program.mapped_inference::@MappedInference_managed) dmaCount([[19, 4], [2, 0]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([1]) actKernelInvocationsCount([1]) mediaCount(0) barrierCount(5) mappedInferenceVersion(@note.MappedInferenceVersion::@MappedInferenceVersion_0_0_0)
        VPUASM.ManagedMappedInference @MappedInference_managed : dmas([[], []]) workItems(@program.workItem::@Enqueue_0_0_0) barrierTasks(@program.managedBarrier::@ConfigureBarrier_0_0_0) bootstrapBarriers() nnrtConfig(@program.nnrt_config::@MappedInference_nnrtConfigManaged) mappedInferenceVersion(@note.MappedInferenceVersion::@MappedInferenceVersion_0_0_0) {actshv_used = 3 : ui8, barrierConfigurationStride = 0 : i64, barrierConfigurationTasksCount = 0 : i64, barrierCount = 5 : i64, barriersReprogrammingCount = 0 : i64, bootstrapBarriersCount = 0 : i64, bootstrapWorkItemsCount = 3 : i64, disableDmaSwFifo = true, dmaCount = [[19, 4], [2, 0]], dma_from_cmx_used = 1 : ui8, dma_from_ddr_used = 3 : ui8, dpu_used = 1 : ui8, final_barrier_id = 4 : i64, media_used = 0 : ui8, workItemsCount = 3 : i64, workloadManagementBarrierProgrammingMode = #VPURegMapped.workload_management_barrier_programming_mode<ALL_BARRIER_DMAS_SCHEDULED>}
      }
    }
    return
  }
}


// Kernel Params
// CHECK: VPUASM.KernelParams @KernelParams_0_0_0
// CHECK-SAME: releaseDesc [@task.dma.0.0::@NNDMA_0_0_17, @task.dma.0.1::@NNDMA_0_1_3]

// CHECK: VPUASM.KernelParams @KernelParams_0_1_0
// CHECK-SAME: releaseDesc [@task.dma.0.0::@NNDMA_0_0_17, @task.dma.0.1::@NNDMA_0_1_3]

// Fetch DMA has input set to SkipDMAs
// CHECK: VPUASM.NNDMA @NNDMA_0_0_10 idx(!VPURegMapped.Index<0:0:10>) links(@task.dma.0.0::@NNDMA_0_0_11) input(@task.dma.0.0::@NNDMA_0_0_15)
// CHECK: VPUASM.NNDMA @NNDMA_0_0_11 idx(!VPURegMapped.Index<0:0:11>) links(@task.dma.0.0::@NNDMA_0_0_12) input(@task.dma.0.1::@NNDMA_0_1_1)
// CHECK: VPUASM.NNDMA @NNDMA_0_0_12 idx(!VPURegMapped.Index<0:0:12>) links(@task.dma.0.0::@NNDMA_0_0_13) input(@task.dma.0.0::@NNDMA_0_0_16)
// CHECK: VPUASM.NNDMA @NNDMA_0_0_13 idx(!VPURegMapped.Index<0:0:13>) links(@task.dma.0.0::@NNDMA_0_0_14) input(@task.dma.0.1::@NNDMA_0_1_2)

// Creating a Loop of Skip in DDR Channel
// CHECK: VPUASM.NNDMA @NNDMA_0_0_15 idx(!VPURegMapped.Index<0:0:15>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_0_0) links(@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_1_0)
// CHECK: VPUASM.NNDMA @NNDMA_0_0_16 idx(!VPURegMapped.Index<0:0:16>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_1_0) links(@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_0_0)

// Creating a loop of Skip in CMX Channel
// CHECK: VPUASM.NNDMA @NNDMA_0_1_1 idx(!VPURegMapped.Index<0:1:1>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_0_1) links(@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_1_1)
// CHECK: VPUASM.NNDMA @NNDMA_0_1_2 idx(!VPURegMapped.Index<0:1:2>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_1_1) links(@program.metadata.cmx::@DeclareTaskBuffer_DMA_0_0_1)
