//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --link-enqueue-targets="workload-management-mode=FWLM_V1_PAGES" %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @DpusAndEnqueueDmaOp attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
  config.PipelineOptions @Options {
    config.Option @config.UseDedicatedFifoPerShaveEngine : false
  }
  config.Resources 1 of @NCE at 1.700000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        config.MemoryResource 512 bytes of @CMX_NN offset 0
      }
    }
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.ExecutorResource 1 of @M2I
  config.ExecutorResource 1 of @DMA_NN
  config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x14x14xf16>
  }
  func.func @main(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<1x16x14x14xf16, @DDR>) -> memref<1x16x14x14xf16, @DDR> {
    %cst = const.Declare memref<1x1x1x4864xui8> = dense<1> : tensor<1x1x1x4864xui8>
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x14x14xf16, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <16896> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <17152> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %10 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <0, -1> -> !VPURegMapped.Index<0:0:0>
    %11 = VPUMI40XX.ConfigureBarrier {consumer_count = 4 : ui8, producer_count = 1 : ui8}(%10 : !VPURegMapped.Index<0:0:0>) <1, -1> -> !VPURegMapped.Index<0:0:1>
    %14 = VPUMI40XX.ConfigureBarrier {consumer_count = 0 : ui8, isFinalBarrier, producer_count = 4 : ui8}(%11 : !VPURegMapped.Index<0:0:1>) <2, -1> -> !VPURegMapped.Index<0:0:2>

    %15 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
    %16 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
    %17 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:2>
    %18 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:3>

    %19 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
    %20 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
    %21 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:2>
    %22 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:3>

    %enq_dma0 = VPUMI40XX.NNDMA {enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 1 : i64>, port = 0 : i64} inputs(%buf0 : memref<0x0x0x0xi32, @DDR>) outputs(%buf1 : memref<0x0x0x0xi32, @DDR>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%11 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %enq_dma1 = VPUMI40XX.NNDMA {enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 2 : i64, endTask = 3 : i64>, port = 0 : i64} inputs(%buf0 : memref<0x0x0x0xi32, @DDR>) outputs(%buf1 : memref<0x0x0x0xi32, @DDR>) previousDMA(%enq_dma0 : !VPURegMapped.Index<0:0:0>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%11 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>

    %23 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%15 : !VPURegMapped.Index<0:0:0>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%14 : !VPURegMapped.Index<0:0:2>) -> <0:0:0> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %24 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%16 : !VPURegMapped.Index<0:0:1>) previousTask(%23 : !VPURegMapped.Index<0:0:0>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%14 : !VPURegMapped.Index<0:0:2>) -> <0:0:1> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %25 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%17 : !VPURegMapped.Index<0:0:2>) previousTask(%24 : !VPURegMapped.Index<0:0:1>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%14 : !VPURegMapped.Index<0:0:2>) -> <0:0:2> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %26 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%18 : !VPURegMapped.Index<0:0:3>) previousTask(%25 : !VPURegMapped.Index<0:0:2>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%14 : !VPURegMapped.Index<0:0:2>) -> <0:0:3> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }

    %27 = VPUMI40XX.DPUVariant taskLocation(%19 : !VPURegMapped.Index<0:0:0>) calls(%23 : <0:0:0>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
    %28 = VPUMI40XX.DPUVariant taskLocation(%20 : !VPURegMapped.Index<0:0:1>) previousTask(%27 : !VPURegMapped.Index<0:0:0>) calls(%24 : <0:0:1>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], lastSecondaryTaskInExecutionGroup, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
    %29 = VPUMI40XX.DPUVariant taskLocation(%21 : !VPURegMapped.Index<0:0:2>) previousTask(%28 : !VPURegMapped.Index<0:0:1>) calls(%25 : <0:0:2>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:2>
    %30 = VPUMI40XX.DPUVariant taskLocation(%22 : !VPURegMapped.Index<0:0:3>) previousTask(%29 : !VPURegMapped.Index<0:0:2>) calls(%26 : <0:0:3>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], lastSecondaryTaskInExecutionGroup, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:3>

    %31 = VPURegMapped.Enqueue (%enq_dma0 -> %enq_dma1 : <0:0:0> -> <0:0:1>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<DMA>}

    %33 = VPUMI40XX.MappedInference dmas((%enq_dma0) : (!VPURegMapped.Index<0:0:0>)) invariants(%23 : !VPURegMapped.Index<0:0:0>) variants(%27 : !VPURegMapped.Index<0:0:0>) barriers(%10 : !VPURegMapped.Index<0:0:0>) workItemTasks(%31 : !VPURegMapped.Index<0:0:0>) dmaCount([[1]]) invariantCount([4]) variantCount([4]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(3) workItemCount(1) bootsrapWorkItemsCount(1) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x14x14xf16, @DDR>
  }
}

//CHECK: [[DMA_ENQ0:%.+]] = VPUMI40XX.NNDMA
//CHECK-SAME: {enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 1 : i64>, port = 0 : i64
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[DMA_ENQ0_IDX:.+]]

//CHECK: [[DMA_ENQ1:%.+]] = VPUMI40XX.NNDMA
//CHECK-SAME: {enqueue_dma_attr = #VPUIP.EnqueueDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 2 : i64, endTask = 3 : i64>, port = 0 : i64
//CHECK-SAME: taskLinkAttrName = #VPURegMapped.IndexType<[[DMA_ENQ0_IDX]]>
//CHECK-SAME: previousDMA([[DMA_ENQ0]] : !VPURegMapped.Index[[DMA_ENQ0_IDX]])
//CHECK-SAME: -> !VPURegMapped.Index[[DMA_ENQ1_IDX:.+]]

//CHECK: VPUMI40XX.DPUInvariant
//CHECK-NOT: taskLinkAttrName

//CHECK: VPUMI40XX.DPUInvariant
//CHECK-NOT: taskLinkAttrName

//CHECK: VPUMI40XX.DPUInvariant
//CHECK-NOT: taskLinkAttrName

//CHECK: VPUMI40XX.DPUInvariant
//CHECK-NOT: taskLinkAttrName

//CHECK: %[[VAR0:.+]] = VPUMI40XX.DPUVariant
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> [[VAR0_IDX:.+]]

//CHECK: %[[VAR1:.+]] = VPUMI40XX.DPUVariant
//CHECK-SAME: taskLinkAttrName = #VPURegMapped.IndexType<[[VAR0_IDX]]>
//CHECK-SAME: -> [[VAR1_IDX:.+]]

//CHECK: %[[VAR2:.+]] = VPUMI40XX.DPUVariant
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> [[VAR2_IDX:.+]]

//CHECK: %[[VAR3:.+]] = VPUMI40XX.DPUVariant
//CHECK-SAME: taskLinkAttrName = #VPURegMapped.IndexType<[[VAR2_IDX]]>
//CHECK-SAME: -> [[VAR3_IDX:.+]]

//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: ([[DMA_ENQ0]] -> [[DMA_ENQ0]] : [[DMA_ENQ0_IDX]] -> [[DMA_ENQ0_IDX]])
