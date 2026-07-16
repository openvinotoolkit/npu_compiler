//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --add-enqueue-ops %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @TwoDmaFifosEnqueueOpsForSameBarrierNotNextToEachOther attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
  config.Resources 2 of @NCE at 1.700000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        config.MemoryResource 512 bytes of @CMX_NN offset 0
      }
    }
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
    DataInfo "output" : tensor<1x16x16x16xf16>
  }
  func.func @main(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<1x16x16x16xf16, @DDR>) -> memref<1x16x16x16xf16, @DDR> {

    %dummy_buf_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x16x16xf16, [@CMX_NN, 0]>
    %dummy_buf_ddr = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x16x16xf16, @DDR>

    %bar0 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}> <0, 3> -> !VPURegMapped.Index<0:0:0>
    %bar1 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}>(%bar0 : !VPURegMapped.Index<0:0:0>) <1, 4> -> !VPURegMapped.Index<0:0:1>
    %bar2 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}>(%bar1 : !VPURegMapped.Index<0:0:1>) <2, 5> -> !VPURegMapped.Index<0:0:2>
    %bar3 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}>(%bar2 : !VPURegMapped.Index<0:0:2>) <0, 6> -> !VPURegMapped.Index<0:0:3>
    %bar4 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}>(%bar3 : !VPURegMapped.Index<0:0:3>) <1, -1> -> !VPURegMapped.Index<0:0:4>
    %bar5 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}>(%bar4 : !VPURegMapped.Index<0:0:4>) <2, -1> -> !VPURegMapped.Index<0:0:5>
    %bar6 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, isFinalBarrier, producer_count = 1 : ui8}>(%bar5 : !VPURegMapped.Index<0:0:5>) <0, -1> -> !VPURegMapped.Index<0:0:6>

    %dma_ddr_0 = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs(%dummy_buf_ddr : memref<1x16x16x16xf16, @DDR>) outputs(%dummy_buf_cmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) waits(%bar0 : !VPURegMapped.Index<0:0:0>) updates(%bar1 : !VPURegMapped.Index<0:0:1>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %dma_ddr_1 = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs(%dummy_buf_ddr : memref<1x16x16x16xf16, @DDR>) outputs(%dummy_buf_cmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) previousDMA(%dma_ddr_0 : !VPURegMapped.Index<0:0:0>) waits(%bar1 : !VPURegMapped.Index<0:0:1>) updates(%bar2 : !VPURegMapped.Index<0:0:2>) start_after(2) clean_after(1) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %dma_ddr_2 = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs(%dummy_buf_ddr : memref<1x16x16x16xf16, @DDR>) outputs(%dummy_buf_cmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) previousDMA(%dma_ddr_1 : !VPURegMapped.Index<0:0:1>) waits(%bar2 : !VPURegMapped.Index<0:0:2>) updates(%bar3 : !VPURegMapped.Index<0:0:3>) enqueueBarrier(%bar0 : !VPURegMapped.Index<0:0:0>) start_after(3) clean_after(2) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %dma_cmx_0 = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs(%dummy_buf_cmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) outputs(%dummy_buf_ddr : memref<1x16x16x16xf16, @DDR>) waits(%bar3 : !VPURegMapped.Index<0:0:3>) updates(%bar4 : !VPURegMapped.Index<0:0:4>) enqueueBarrier(%bar1 : !VPURegMapped.Index<0:0:1>) start_after(4) clean_after(3) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %dma_cmx_1 = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs(%dummy_buf_cmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) outputs(%dummy_buf_ddr : memref<1x16x16x16xf16, @DDR>) previousDMA(%dma_cmx_0 : !VPURegMapped.Index<0:1:0>) waits(%bar4 : !VPURegMapped.Index<0:0:4>) updates(%bar5 : !VPURegMapped.Index<0:0:5>) enqueueBarrier(%bar0 : !VPURegMapped.Index<0:0:0>) start_after(5) clean_after(4) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:1>
    %dma_ddr_3 = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs(%dummy_buf_ddr : memref<1x16x16x16xf16, @DDR>) outputs(%dummy_buf_cmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) previousDMA(%dma_ddr_2 : !VPURegMapped.Index<0:0:2>) waits(%bar5 : !VPURegMapped.Index<0:0:5>) updates(%bar6 : !VPURegMapped.Index<0:0:6>) enqueueBarrier(%bar2 : !VPURegMapped.Index<0:0:2>) start_after(6) clean_after(5) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>

    %30 = VPUMI40XX.MappedInference dmas((%dma_ddr_0, %dma_cmx_0) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) barriers(%bar0 : !VPURegMapped.Index<0:0:0>) dmaCount([[4, 2]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(7) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x16x16xf16, @DDR>
  }
}

//CHECK: [[BAR0:%.+]] = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}> <0, 3> -> !VPURegMapped.Index<0:0:0>
//CHECK: [[BAR1:%.+]] = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}>([[BAR0]] : !VPURegMapped.Index<0:0:0>) <1, 4> -> !VPURegMapped.Index<0:0:1>
//CHECK: [[BAR2:%.+]] = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}>([[BAR1]] : !VPURegMapped.Index<0:0:1>) <2, 5> -> !VPURegMapped.Index<0:0:2>
//CHECK: [[BAR3:%.+]] = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}>([[BAR2]] : !VPURegMapped.Index<0:0:2>) <0, 6> -> !VPURegMapped.Index<0:0:3>
//CHECK: [[BAR4:%.+]] = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}>([[BAR3]] : !VPURegMapped.Index<0:0:3>) <1, -1> -> !VPURegMapped.Index<0:0:4>
//CHECK: [[BAR5:%.+]] = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}>([[BAR4]] : !VPURegMapped.Index<0:0:4>) <2, -1> -> !VPURegMapped.Index<0:0:5>
//CHECK: [[BAR6:%.+]] = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, isFinalBarrier, producer_count = 1 : ui8}>([[BAR5]] : !VPURegMapped.Index<0:0:5>) <0, -1> -> !VPURegMapped.Index<0:0:6>

//CHECK: [[DMA_DDR_0:%.+]] = VPUMI40XX.NNDMA
//CHECK: [[DMA_DDR_1:%.+]] = VPUMI40XX.NNDMA
//CHECK: [[DMA_DDR_2:%.+]] = VPUMI40XX.NNDMA
//CHECK-SAME: enqueueBarrier([[BAR0]] : !VPURegMapped.Index<0:0:0>)
//CHECK: [[DMA_CMX_0:%.+]] = VPUMI40XX.NNDMA
//CHECK-SAME: enqueueBarrier([[BAR1]] : !VPURegMapped.Index<0:0:1>)
//CHECK: [[DMA_CMX_1:%.+]] = VPUMI40XX.NNDMA
//CHECK-SAME: enqueueBarrier([[BAR0]] : !VPURegMapped.Index<0:0:0>)
//CHECK: [[DMA_DDR_3:%.+]] = VPUMI40XX.NNDMA
//CHECK-SAME: enqueueBarrier([[BAR2]] : !VPURegMapped.Index<0:0:2>)

// Resulting enqueue order: Enq(BAR1) -> Enq(BAR0) -> Enq(BAR0) -> Enq(BAR2) does not
// follow barrier index but is a possible enqueue ordering that guarantees correct execution and aligns
// with WorkItem ordering restrictions
//CHECK: [[ENQ0:%.+]] = VPURegMapped.Enqueue at([[BAR1]] : !VPURegMapped.Index<0:0:1>) ([[DMA_CMX_0]] -> [[DMA_CMX_0]] : <0:1:0> -> <0:1:0>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<DMA>}
//CHECK: [[ENQ1:%.+]] = VPURegMapped.Enqueue previousTaskIdx([[ENQ0]] : !VPURegMapped.Index<0:0:0>) at([[BAR0]] : !VPURegMapped.Index<0:0:0>) ([[DMA_DDR_2]] -> [[DMA_DDR_2]] : <0:0:2> -> <0:0:2>) -> !VPURegMapped.Index<0:0:1> {taskType = #VPURegMapped.task_type<DMA>}
//CHECK: [[ENQ2:%.+]] = VPURegMapped.Enqueue previousTaskIdx([[ENQ1]] : !VPURegMapped.Index<0:0:1>) at([[BAR0]] : !VPURegMapped.Index<0:0:0>) ([[DMA_CMX_1]] -> [[DMA_CMX_1]] : <0:1:1> -> <0:1:1>) -> !VPURegMapped.Index<0:0:2> {taskType = #VPURegMapped.task_type<DMA>}
//CHECK: [[ENQ3:%.+]] = VPURegMapped.Enqueue previousTaskIdx([[ENQ2]] : !VPURegMapped.Index<0:0:2>) at([[BAR2]] : !VPURegMapped.Index<0:0:2>) ([[DMA_DDR_3]] -> [[DMA_DDR_3]] : <0:0:3> -> <0:0:3>) -> !VPURegMapped.Index<0:0:3> {taskType = #VPURegMapped.task_type<DMA>}

//CHECK: workItemTasks([[ENQ0]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: workItemCount(4)
