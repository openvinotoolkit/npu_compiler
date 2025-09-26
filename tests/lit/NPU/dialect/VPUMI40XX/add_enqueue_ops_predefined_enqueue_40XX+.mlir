//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --add-enqueue-ops="enable-predefined-enqueue=true" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @Convolution attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
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
    %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x16x16xf16, @DDR>
    %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x14x14xf16, @DDR>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x14x14xf16, [@CMX_NN, 0]>
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <16896> -> memref<1x1x1x4864xui8, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <16896> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <17152> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %10 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <4, -1> -> !VPURegMapped.Index<0:0:0>
    %11 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}(%10 : !VPURegMapped.Index<0:0:0>) <0, -1> -> !VPURegMapped.Index<0:0:1>
    %12 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 2 : ui8}(%11 : !VPURegMapped.Index<0:0:1>) <1, -1> -> !VPURegMapped.Index<0:0:2>
    %13 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}(%12 : !VPURegMapped.Index<0:0:2>) <2, -1> -> !VPURegMapped.Index<0:0:3>
    %14 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, isFinalBarrier, producer_count = 1 : ui8}(%13 : !VPURegMapped.Index<0:0:3>) <3, -1> -> !VPURegMapped.Index<0:0:4>
    %15 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
    %16 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
    %17 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
    %18 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
    %19 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%15 : !VPURegMapped.Index<0:0:0>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%12 : !VPURegMapped.Index<0:0:2>) enqueueBarrier(%10 : !VPURegMapped.Index<0:0:0>) -> <0:0:0> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %20 = VPUMI40XX.DPUInvariant {clean_after = 3 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 4 : ui64} taskLocation(%16 : !VPURegMapped.Index<0:0:1>) previousTask(%19 : !VPURegMapped.Index<0:0:0>) input(%7 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) waits(%12 : !VPURegMapped.Index<0:0:2>) updates(%13 : !VPURegMapped.Index<0:0:3>) enqueueBarrier(%10 : !VPURegMapped.Index<0:0:0>) -> <0:0:1> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %21 = VPUMI40XX.DPUVariant taskLocation(%17 : !VPURegMapped.Index<0:0:0>) calls(%19 : <0:0:0>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
    %22 = VPUMI40XX.DPUVariant taskLocation(%18 : !VPURegMapped.Index<0:0:1>) previousTask(%21 : !VPURegMapped.Index<0:0:0>) calls(%20 : <0:0:1>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
    %23 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %24 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %25 = VPURegMapped.FetchTask primary(%19 -> %20) secondary(%21 -> %22) (<0:0:0> -> <0:0:1> : !VPURegMapped.Index<0:0:0> -> !VPURegMapped.Index<0:0:1>) -> <0:0:0>
    %26 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 0 : i32, srcWidth = 0 : i32, srcStride = 0 : i32, srcPlaneStride = 0 : i32, dstWidth = 0 : i32, dstStride = 0 : i32, dstPlaneStride = 0 : i32>, port = 0 : i64} inputs(%23 : memref<1x1x1x1xi32, @DDR>) outputs(%24 : memref<1x1x1x1xi32, @DDR>) previousDMA(%25 : !VPURegMapped.Index<0:0:0>) updates(%10 : !VPURegMapped.Index<0:0:0>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %27 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x16x16x16xf16, @DDR>) outputs(%2 : memref<1x16x16x16xf16, [@CMX_NN, 0]>) previousDMA(%26 : !VPURegMapped.Index<0:0:1>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%11 : !VPURegMapped.Index<0:0:1>) start_after(2) clean_after(1) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %28 = VPUMI40XX.NNDMA {port = 0 : i64, is_out_of_order} inputs(%cst : memref<1x1x1x4864xui8>) outputs(%5 : memref<1x1x1x4864xui8, [@CMX_NN, 0]>) previousDMA(%27 : !VPURegMapped.Index<0:0:2>) updates(%12 : !VPURegMapped.Index<0:0:2>) start_after(3) clean_after(2) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>
    %29 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x16x14x14xf16, @DDR>) waits(%13 : !VPURegMapped.Index<0:0:3>) updates(%14 : !VPURegMapped.Index<0:0:4>) start_after(5) clean_after(4) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %30 = VPUMI40XX.MappedInference dmas((%25, %29) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) invariants(%19 : !VPURegMapped.Index<0:0:0>) variants(%21 : !VPURegMapped.Index<0:0:0>) barriers(%10 : !VPURegMapped.Index<0:0:0>) dmaCount([[4, 1]]) invariantCount([2]) variantCount([2]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(5) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x14x14xf16, @DDR>
  }
}

//CHECK: [[VAL10:%.+]] = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <4, -1> -> !VPURegMapped.Index<0:0:0>
//CHECK: [[VAL21:%.+]] = VPUMI40XX.DPUVariant
//CHECK: [[VAL22:%.+]] = VPUMI40XX.DPUVariant
//CHECK: [[VAL30:%.+]] = VPURegMapped.Enqueue at([[VAL10]] : !VPURegMapped.Index<0:0:0>) ([[VAL21]] -> [[VAL22]] : <0:0:0> -> <0:0:1>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<DPUVariant>}
//CHECK: workItemTasks([[VAL30]] : !VPURegMapped.Index<0:0:0>)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @TwoDmaFifosEnqueueOpsForSameBarrierNotNextToEachOther attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
  config.Resources 2 of @NCE at 1.700000e+03 MHz {
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
    DataInfo "output" : tensor<1x16x16x16xf16>
  }
  func.func @main(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<1x16x16x16xf16, @DDR>) -> memref<1x16x16x16xf16, @DDR> {

    %dummy_buf_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x16x16xf16, [@CMX_NN, 0]>
    %dummy_buf_ddr = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x16x16xf16, @DDR>

    %bar0 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <0, 3> -> !VPURegMapped.Index<0:0:0>
    %bar1 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}(%bar0 : !VPURegMapped.Index<0:0:0>) <1, 4> -> !VPURegMapped.Index<0:0:1>
    %bar2 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}(%bar1 : !VPURegMapped.Index<0:0:1>) <2, 5> -> !VPURegMapped.Index<0:0:2>
    %bar3 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}(%bar2 : !VPURegMapped.Index<0:0:2>) <0, 6> -> !VPURegMapped.Index<0:0:3>
    %bar4 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}(%bar3 : !VPURegMapped.Index<0:0:3>) <1, -1> -> !VPURegMapped.Index<0:0:4>
    %bar5 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}(%bar4 : !VPURegMapped.Index<0:0:4>) <2, -1> -> !VPURegMapped.Index<0:0:5>
    %bar6 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, isFinalBarrier, producer_count = 1 : ui8}(%bar5 : !VPURegMapped.Index<0:0:5>) <0, -1> -> !VPURegMapped.Index<0:0:6>

    %dma_ddr_0 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%dummy_buf_ddr : memref<1x16x16x16xf16, @DDR>) outputs(%dummy_buf_cmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) waits(%bar0 : !VPURegMapped.Index<0:0:0>) updates(%bar1 : !VPURegMapped.Index<0:0:1>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %dma_ddr_1 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%dummy_buf_ddr : memref<1x16x16x16xf16, @DDR>) outputs(%dummy_buf_cmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) previousDMA(%dma_ddr_0 : !VPURegMapped.Index<0:0:0>) waits(%bar1 : !VPURegMapped.Index<0:0:1>) updates(%bar2 : !VPURegMapped.Index<0:0:2>) start_after(2) clean_after(1) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %dma_ddr_2 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%dummy_buf_ddr : memref<1x16x16x16xf16, @DDR>) outputs(%dummy_buf_cmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) previousDMA(%dma_ddr_1 : !VPURegMapped.Index<0:0:1>) waits(%bar2 : !VPURegMapped.Index<0:0:2>) updates(%bar3 : !VPURegMapped.Index<0:0:3>) enqueueBarrier(%bar0 : !VPURegMapped.Index<0:0:0>) start_after(3) clean_after(2) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %dma_cmx_0 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%dummy_buf_cmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) outputs(%dummy_buf_ddr : memref<1x16x16x16xf16, @DDR>) waits(%bar3 : !VPURegMapped.Index<0:0:3>) updates(%bar4 : !VPURegMapped.Index<0:0:4>) enqueueBarrier(%bar1 : !VPURegMapped.Index<0:0:1>) start_after(4) clean_after(3) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %dma_cmx_1 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%dummy_buf_cmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) outputs(%dummy_buf_ddr : memref<1x16x16x16xf16, @DDR>) previousDMA(%dma_cmx_0 : !VPURegMapped.Index<0:1:0>) waits(%bar4 : !VPURegMapped.Index<0:0:4>) updates(%bar5 : !VPURegMapped.Index<0:0:5>) enqueueBarrier(%bar0 : !VPURegMapped.Index<0:0:0>) start_after(5) clean_after(4) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:1>
    %dma_ddr_3 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%dummy_buf_ddr : memref<1x16x16x16xf16, @DDR>) outputs(%dummy_buf_cmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) previousDMA(%dma_ddr_2 : !VPURegMapped.Index<0:0:2>) waits(%bar5 : !VPURegMapped.Index<0:0:5>) updates(%bar6 : !VPURegMapped.Index<0:0:6>) enqueueBarrier(%bar2 : !VPURegMapped.Index<0:0:2>) start_after(6) clean_after(5) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>

    %30 = VPUMI40XX.MappedInference dmas((%dma_ddr_0, %dma_cmx_0) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) barriers(%bar0 : !VPURegMapped.Index<0:0:0>) dmaCount([[4, 2]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(7) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x16x16xf16, @DDR>
  }
}

//CHECK: [[BAR0:%.+]] = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <0, 3> -> !VPURegMapped.Index<0:0:0>
//CHECK: [[BAR1:%.+]] = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}([[BAR0]] : !VPURegMapped.Index<0:0:0>) <1, 4> -> !VPURegMapped.Index<0:0:1>
//CHECK: [[BAR2:%.+]] = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}([[BAR1]] : !VPURegMapped.Index<0:0:1>) <2, 5> -> !VPURegMapped.Index<0:0:2>
//CHECK: [[BAR3:%.+]] = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}([[BAR2]] : !VPURegMapped.Index<0:0:2>) <0, 6> -> !VPURegMapped.Index<0:0:3>
//CHECK: [[BAR4:%.+]] = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}([[BAR3]] : !VPURegMapped.Index<0:0:3>) <1, -1> -> !VPURegMapped.Index<0:0:4>
//CHECK: [[BAR5:%.+]] = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}([[BAR4]] : !VPURegMapped.Index<0:0:4>) <2, -1> -> !VPURegMapped.Index<0:0:5>
//CHECK: [[BAR6:%.+]] = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, isFinalBarrier, producer_count = 1 : ui8}([[BAR5]] : !VPURegMapped.Index<0:0:5>) <0, -1> -> !VPURegMapped.Index<0:0:6>

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

//CHECK: [[ENQ0:%.+]] = VPURegMapped.Enqueue at([[BAR0]] : !VPURegMapped.Index<0:0:0>) ([[DMA_DDR_2]] -> [[DMA_DDR_2]] : <0:0:2> -> <0:0:2>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<DMA>}
//CHECK: [[ENQ1:%.+]] = VPURegMapped.Enqueue previousTaskIdx([[ENQ0]] : !VPURegMapped.Index<0:0:0>) at([[BAR1]] : !VPURegMapped.Index<0:0:1>) ([[DMA_CMX_0]] -> [[DMA_CMX_0]] : <0:1:0> -> <0:1:0>) -> !VPURegMapped.Index<0:0:1> {taskType = #VPURegMapped.task_type<DMA>}
//CHECK: [[ENQ2:%.+]] = VPURegMapped.Enqueue previousTaskIdx([[ENQ1]] : !VPURegMapped.Index<0:0:1>) at([[BAR0]] : !VPURegMapped.Index<0:0:0>) ([[DMA_CMX_1]] -> [[DMA_CMX_1]] : <0:1:1> -> <0:1:1>) -> !VPURegMapped.Index<0:0:2> {taskType = #VPURegMapped.task_type<DMA>}
//CHECK: [[ENQ3:%.+]] = VPURegMapped.Enqueue previousTaskIdx([[ENQ2]] : !VPURegMapped.Index<0:0:2>) at([[BAR2]] : !VPURegMapped.Index<0:0:2>) ([[DMA_DDR_3]] -> [[DMA_DDR_3]] : <0:0:3> -> <0:0:3>) -> !VPURegMapped.Index<0:0:3> {taskType = #VPURegMapped.task_type<DMA>}

//CHECK: workItemTasks([[ENQ0]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: workItemCount(4)


// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @Softmax attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
  config.Resources 2 of @NCE at 1.700000e+03 MHz {
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
    DataInfo "input" : tensor<1x1000x1x1xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1000x1x1xf16>
  }
  func.func @main(%arg0: memref<1x1000x1x1xf16, @DDR>, %arg1: memref<1x1000x1x1xf16, @DDR>) -> memref<1x1000x1x1xf16, @DDR> {
  %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
  %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1000x1x1xf16, @DDR>
  %2 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
  %3 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
  %4 = VPURT.DeclareBuffer <CMX_NN> [0] <1473536> -> memref<16xui32, [@CMX_NN, 0]>
  %5 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %6 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %7 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %8 = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %9 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %10 = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %11 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %12 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %13 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %14 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %15 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %16 = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %17 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %18 = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %19 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %20 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %21 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1000x1x1xf16, [@CMX_NN, 0]>
  %22 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %23 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  // buffers for KernelParams of ActKernelInvocation ops in tile 0, list 1:
  %329 = VPURT.DeclareBuffer <CMX_NN> [0] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %330 = VPURT.DeclareBuffer <CMX_NN> [0] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %331 = VPURT.DeclareBuffer <CMX_NN> [0] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %332 = VPURT.DeclareBuffer <CMX_NN> [0] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %333 = VPURT.DeclareBuffer <CMX_NN> [0] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %334 = VPURT.DeclareBuffer <CMX_NN> [0] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %335 = VPURT.DeclareBuffer <CMX_NN> [0] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %336 = VPURT.DeclareBuffer <CMX_NN> [0] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  // buffers for KernelParams of ActKernelInvocation ops in tile 1, list 1:
  %337 = VPURT.DeclareBuffer <CMX_NN> [1] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %338 = VPURT.DeclareBuffer <CMX_NN> [1] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %339 = VPURT.DeclareBuffer <CMX_NN> [1] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %340 = VPURT.DeclareBuffer <CMX_NN> [1] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %341 = VPURT.DeclareBuffer <CMX_NN> [1] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %342 = VPURT.DeclareBuffer <CMX_NN> [1] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %343 = VPURT.DeclareBuffer <CMX_NN> [1] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %344 = VPURT.DeclareBuffer <CMX_NN> [1] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>

  %345 = VPURT.DeclareBuffer <CMX_NN> [0] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %346 = VPURT.DeclareBuffer <CMX_NN> [1] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>

  %24 = VPUMI40XX.DeclareKernelText kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
  %25 = VPUMI40XX.DeclareKernelEntry kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
  %26 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
  %347 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:1:0>
  %27 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:0:0>
  %348 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:1:0>
  %28 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:1>
  %349 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:1:1>
  %29 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:0:1>
  %350 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:1:1>
  %30 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:2>
  %351 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:1:2>
  %31 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:0:2>
  %352 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:1:2>
  %32 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:3>
  %353 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:1:3>
  %33 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:0:3>
  %354 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:1:3>
  %34 = VPUMI40XX.KernelParams inputs(%5 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%9 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:0:0>
  %355 = VPUMI40XX.KernelParams inputs(%329 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%330 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:1:0>
  %35 = VPUMI40XX.KernelParams inputs(%6 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%10 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:0:0>
  %356 = VPUMI40XX.KernelParams inputs(%337 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%338 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:1:0>
  %36 = VPUMI40XX.KernelParams inputs(%7 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%13 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:0:1>
  %357 = VPUMI40XX.KernelParams inputs(%331 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%332 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:1:1>
  %37 = VPUMI40XX.KernelParams inputs(%8 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%14 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:0:1>
  %358 = VPUMI40XX.KernelParams inputs(%339 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%340 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:1:1>
  %38 = VPUMI40XX.KernelParams inputs(%11 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%17 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:0:2>
  %359 = VPUMI40XX.KernelParams inputs(%333 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%334 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:1:2>
  %39 = VPUMI40XX.KernelParams inputs(%12 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%18 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:0:2>
  %360 = VPUMI40XX.KernelParams inputs(%341 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%342 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:1:2>
  %40 = VPUMI40XX.KernelParams inputs(%15 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%19 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:0:3>
  %361 = VPUMI40XX.KernelParams inputs(%335 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%336 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:1:3>
  %41 = VPUMI40XX.KernelParams inputs(%16 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%20 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:0:3>
  %362 = VPUMI40XX.KernelParams inputs(%343 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%344 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:1:3>
  %42 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, isStartBarrier, producer_count = 1 : ui8} <0, -1> -> !VPURegMapped.Index<0:0:0>
  %43 = VPUMI40XX.ConfigureBarrier {consumer_count = 4 : ui8, producer_count = 1 : ui8}(%42 : !VPURegMapped.Index<0:0:0>) <1, -1> -> !VPURegMapped.Index<0:0:1>
  %44 = VPUMI40XX.ConfigureBarrier {consumer_count = 4 : ui8, producer_count = 4 : ui8}(%43 : !VPURegMapped.Index<0:0:1>) <2, -1> -> !VPURegMapped.Index<0:0:2>
  %45 = VPUMI40XX.ConfigureBarrier {consumer_count = 4 : ui8, producer_count = 4 : ui8}(%44 : !VPURegMapped.Index<0:0:2>) <3, -1> -> !VPURegMapped.Index<0:0:3>
  %46 = VPUMI40XX.ConfigureBarrier {consumer_count = 4 : ui8, producer_count = 4 : ui8}(%45 : !VPURegMapped.Index<0:0:3>) <4, -1> -> !VPURegMapped.Index<0:0:4>
  %47 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 4 : ui8}(%46 : !VPURegMapped.Index<0:0:4>) <5, -1> -> !VPURegMapped.Index<0:0:5>
  %48 = VPUMI40XX.ConfigureBarrier {consumer_count = 0 : ui8, isFinalBarrier, producer_count = 1 : ui8}(%47 : !VPURegMapped.Index<0:0:5>) <6, -1> -> !VPURegMapped.Index<0:0:6>
  %77 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:12>
  %78 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:13>
  %79 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:14>
  %80 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:15>
  %141 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:0:12>
  %142 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:0:13>
  %143 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:0:14>
  %144 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:0:15>
  %177 = VPUMI40XX.ActKernelRange taskLocation(%141 : !VPURegMapped.Index<0:0:12>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%26 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
  %178 = VPUMI40XX.ActKernelRange taskLocation(%142 : !VPURegMapped.Index<0:0:13>) previousTask(%177 : !VPURegMapped.Index<0:0:0>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%28 : !VPURegMapped.Index<0:0:1>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:1>
  %179 = VPUMI40XX.ActKernelRange taskLocation(%143 : !VPURegMapped.Index<0:0:14>) previousTask(%178 : !VPURegMapped.Index<0:0:1>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%30 : !VPURegMapped.Index<0:0:2>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:2>
  %180 = VPUMI40XX.ActKernelRange taskLocation(%144 : !VPURegMapped.Index<0:0:15>) previousTask(%179 : !VPURegMapped.Index<0:0:2>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%32 : !VPURegMapped.Index<0:0:3>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:3>
  %181 = VPUMI40XX.ActKernelInvocation taskLocation(%77 : !VPURegMapped.Index<0:0:12>) range_index(%177 : <0:0:0>) kernel_params(%34 : <0:0:0>) waits(%43 : !VPURegMapped.Index<0:0:1>) updates(%44 : !VPURegMapped.Index<0:0:2>) enqueueBarrier(%42 : !VPURegMapped.Index<0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
  %182 = VPUMI40XX.ActKernelInvocation taskLocation(%78 : !VPURegMapped.Index<0:0:13>) previousTask(%181 : !VPURegMapped.Index<0:0:0>) range_index(%178 : <0:0:1>) kernel_params(%36 : <0:0:1>) waits(%44 : !VPURegMapped.Index<0:0:2>) updates(%45 : !VPURegMapped.Index<0:0:3>) enqueueBarrier(%42 : !VPURegMapped.Index<0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
  %183 = VPUMI40XX.ActKernelInvocation taskLocation(%79 : !VPURegMapped.Index<0:0:14>) previousTask(%182 : !VPURegMapped.Index<0:0:1>) range_index(%179 : <0:0:2>) kernel_params(%38 : <0:0:2>) waits(%45 : !VPURegMapped.Index<0:0:3>) updates(%46 : !VPURegMapped.Index<0:0:4>) enqueueBarrier(%42 : !VPURegMapped.Index<0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>
  %184 = VPUMI40XX.ActKernelInvocation {lastSecondaryTaskInExecutionGroup} taskLocation(%80 : !VPURegMapped.Index<0:0:15>) previousTask(%183 : !VPURegMapped.Index<0:0:2>) range_index(%180 : <0:0:3>) kernel_params(%40 : <0:0:3>) waits(%46 : !VPURegMapped.Index<0:0:4>) updates(%47 : !VPURegMapped.Index<0:0:5>) enqueueBarrier(%42 : !VPURegMapped.Index<0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:3>
  %363 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:1:12>
  %364 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:1:13>
  %365 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:1:14>
  %366 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:1:15>
  %367 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:1:12>
  %368 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:1:13>
  %369 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:1:14>
  %370 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:1:15>
  %371 = VPUMI40XX.ActKernelRange taskLocation(%367 : !VPURegMapped.Index<0:1:12>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%347 : !VPURegMapped.Index<0:1:0>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:1:0>
  %372 = VPUMI40XX.ActKernelRange taskLocation(%368 : !VPURegMapped.Index<0:1:13>) previousTask(%371 : !VPURegMapped.Index<0:1:0>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%349 : !VPURegMapped.Index<0:1:1>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:1:1>
  %373 = VPUMI40XX.ActKernelRange taskLocation(%369 : !VPURegMapped.Index<0:1:14>) previousTask(%372 : !VPURegMapped.Index<0:1:1>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%351 : !VPURegMapped.Index<0:1:2>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:1:2>
  %374 = VPUMI40XX.ActKernelRange taskLocation(%370 : !VPURegMapped.Index<0:1:15>) previousTask(%373 : !VPURegMapped.Index<0:1:2>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%353 : !VPURegMapped.Index<0:1:3>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:1:3>
  %375 = VPUMI40XX.ActKernelInvocation taskLocation(%363 : !VPURegMapped.Index<0:1:12>) range_index(%371 : <0:1:0>) kernel_params(%355 : <0:1:0>) waits(%43 : !VPURegMapped.Index<0:0:1>) updates(%44 : !VPURegMapped.Index<0:0:2>) enqueueBarrier(%42 : !VPURegMapped.Index<0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:1:0>
  %376 = VPUMI40XX.ActKernelInvocation taskLocation(%364 : !VPURegMapped.Index<0:1:13>) previousTask(%375 : !VPURegMapped.Index<0:1:0>) range_index(%372 : <0:1:1>) kernel_params(%357 : <0:1:1>) waits(%44 : !VPURegMapped.Index<0:0:2>) updates(%45 : !VPURegMapped.Index<0:0:3>) enqueueBarrier(%42 : !VPURegMapped.Index<0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:1:1>
  %377 = VPUMI40XX.ActKernelInvocation taskLocation(%365 : !VPURegMapped.Index<0:1:14>) previousTask(%376 : !VPURegMapped.Index<0:1:1>) range_index(%373 : <0:1:2>) kernel_params(%359 : <0:1:2>) waits(%45 : !VPURegMapped.Index<0:0:3>) updates(%46 : !VPURegMapped.Index<0:0:4>) enqueueBarrier(%42 : !VPURegMapped.Index<0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:1:2>
  %378 = VPUMI40XX.ActKernelInvocation {lastSecondaryTaskInExecutionGroup} taskLocation(%366 : !VPURegMapped.Index<0:1:15>) previousTask(%377 : !VPURegMapped.Index<0:1:2>) range_index(%374 : <0:1:3>) kernel_params(%361 : <0:1:3>) waits(%46 : !VPURegMapped.Index<0:0:4>) updates(%47 : !VPURegMapped.Index<0:0:5>) enqueueBarrier(%42 : !VPURegMapped.Index<0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:1:3>
  %213 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<1:0:12>
  %214 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<1:0:13>
  %215 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<1:0:14>
  %216 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<1:0:15>
  %277 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<1:0:12>
  %278 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<1:0:13>
  %279 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<1:0:14>
  %280 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<1:0:15>
  %313 = VPUMI40XX.ActKernelRange taskLocation(%277 : !VPURegMapped.Index<1:0:12>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%27 : !VPURegMapped.Index<1:0:0>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:0:0>
  %314 = VPUMI40XX.ActKernelRange taskLocation(%278 : !VPURegMapped.Index<1:0:13>) previousTask(%313 : !VPURegMapped.Index<1:0:0>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%29 : !VPURegMapped.Index<1:0:1>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:0:1>
  %315 = VPUMI40XX.ActKernelRange taskLocation(%279 : !VPURegMapped.Index<1:0:14>) previousTask(%314 : !VPURegMapped.Index<1:0:1>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%31 : !VPURegMapped.Index<1:0:2>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:0:2>
  %316 = VPUMI40XX.ActKernelRange taskLocation(%280 : !VPURegMapped.Index<1:0:15>) previousTask(%315 : !VPURegMapped.Index<1:0:2>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%33 : !VPURegMapped.Index<1:0:3>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:0:3>
  %317 = VPUMI40XX.ActKernelInvocation taskLocation(%213 : !VPURegMapped.Index<1:0:12>) range_index(%313 : <1:0:0>) kernel_params(%35 : <1:0:0>) waits(%43 : !VPURegMapped.Index<0:0:1>) updates(%44 : !VPURegMapped.Index<0:0:2>) enqueueBarrier(%42 : !VPURegMapped.Index<0:0:0>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:0>
  %318 = VPUMI40XX.ActKernelInvocation taskLocation(%214 : !VPURegMapped.Index<1:0:13>) previousTask(%317 : !VPURegMapped.Index<1:0:0>) range_index(%314 : <1:0:1>) kernel_params(%37 : <1:0:1>) waits(%44 : !VPURegMapped.Index<0:0:2>) updates(%45 : !VPURegMapped.Index<0:0:3>) enqueueBarrier(%42 : !VPURegMapped.Index<0:0:0>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:1>
  %319 = VPUMI40XX.ActKernelInvocation taskLocation(%215 : !VPURegMapped.Index<1:0:14>) previousTask(%318 : !VPURegMapped.Index<1:0:1>) range_index(%315 : <1:0:2>) kernel_params(%39 : <1:0:2>) waits(%45 : !VPURegMapped.Index<0:0:3>) updates(%46 : !VPURegMapped.Index<0:0:4>) enqueueBarrier(%42 : !VPURegMapped.Index<0:0:0>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:2>
  %320 = VPUMI40XX.ActKernelInvocation {lastSecondaryTaskInExecutionGroup} taskLocation(%216 : !VPURegMapped.Index<1:0:15>) previousTask(%319 : !VPURegMapped.Index<1:0:2>) range_index(%316 : <1:0:3>) kernel_params(%41 : <1:0:3>) waits(%46 : !VPURegMapped.Index<0:0:4>) updates(%47 : !VPURegMapped.Index<0:0:5>) enqueueBarrier(%42 : !VPURegMapped.Index<0:0:0>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:3>
  %379 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<1:1:12>
  %380 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<1:1:13>
  %381 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<1:1:14>
  %382 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<1:1:15>
  %383 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<1:1:12>
  %384 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<1:1:13>
  %385 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<1:1:14>
  %386 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<1:1:15>
  %387 = VPUMI40XX.ActKernelRange taskLocation(%383 : !VPURegMapped.Index<1:1:12>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%348 : !VPURegMapped.Index<1:1:0>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:1:0>
  %388 = VPUMI40XX.ActKernelRange taskLocation(%384 : !VPURegMapped.Index<1:1:13>) previousTask(%387 : !VPURegMapped.Index<1:1:0>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%350 : !VPURegMapped.Index<1:1:1>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:1:1>
  %389 = VPUMI40XX.ActKernelRange taskLocation(%385 : !VPURegMapped.Index<1:1:14>) previousTask(%388 : !VPURegMapped.Index<1:1:1>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%352 : !VPURegMapped.Index<1:1:2>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:1:2>
  %390 = VPUMI40XX.ActKernelRange taskLocation(%386 : !VPURegMapped.Index<1:1:15>) previousTask(%389 : !VPURegMapped.Index<1:1:2>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%354 : !VPURegMapped.Index<1:1:3>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:1:3>
  %391 = VPUMI40XX.ActKernelInvocation taskLocation(%379 : !VPURegMapped.Index<1:1:12>) range_index(%387 : <1:1:0>) kernel_params(%356 : <1:1:0>) waits(%43 : !VPURegMapped.Index<0:0:1>) updates(%44 : !VPURegMapped.Index<0:0:2>) enqueueBarrier(%42 : !VPURegMapped.Index<0:0:0>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:1:0>
  %392 = VPUMI40XX.ActKernelInvocation taskLocation(%380 : !VPURegMapped.Index<1:1:13>) previousTask(%391 : !VPURegMapped.Index<1:1:0>) range_index(%388 : <1:1:1>) kernel_params(%358 : <1:1:1>) waits(%44 : !VPURegMapped.Index<0:0:2>) updates(%45 : !VPURegMapped.Index<0:0:3>) enqueueBarrier(%42 : !VPURegMapped.Index<0:0:0>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:1:1>
  %393 = VPUMI40XX.ActKernelInvocation taskLocation(%381 : !VPURegMapped.Index<1:1:14>) previousTask(%392 : !VPURegMapped.Index<1:1:1>) range_index(%389 : <1:1:2>) kernel_params(%360 : <1:1:2>) waits(%45 : !VPURegMapped.Index<0:0:3>) updates(%46 : !VPURegMapped.Index<0:0:4>) enqueueBarrier(%42 : !VPURegMapped.Index<0:0:0>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:1:2>
  %394 = VPUMI40XX.ActKernelInvocation {lastSecondaryTaskInExecutionGroup} taskLocation(%382 : !VPURegMapped.Index<1:1:15>) previousTask(%393 : !VPURegMapped.Index<1:1:2>) range_index(%390 : <1:1:3>) kernel_params(%362 : <1:1:3>) waits(%46 : !VPURegMapped.Index<0:0:4>) updates(%47 : !VPURegMapped.Index<0:0:5>) enqueueBarrier(%42 : !VPURegMapped.Index<0:0:0>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:1:3>
  %ft0 = VPURegMapped.FetchTask primary(%387 -> %390) secondary(%391 -> %394) (<1:1:0> -> <1:1:3> : !VPURegMapped.Index<1:1:0> -> !VPURegMapped.Index<1:1:3>) -> <0:0:0> {associated_execution_group_index = 0 : ui64, associated_task_type = #VPURegMapped.task_type<ActKernelRange>, associated_tile_index = 1 : ui64}
  %ft1 = VPURegMapped.FetchTask previousTask(%ft0 : !VPURegMapped.Index<0:0:0>) primary(%313 -> %316) secondary(%317 -> %320) (<1:0:0> -> <1:0:3> : !VPURegMapped.Index<1:0:0> -> !VPURegMapped.Index<1:0:3>) -> <0:0:1> {associated_execution_group_index = 0 : ui64, associated_task_type = #VPURegMapped.task_type<ActKernelRange>, associated_tile_index = 1 : ui64}
  %ft2 = VPURegMapped.FetchTask previousTask(%ft1 : !VPURegMapped.Index<0:0:1>) primary(%371 -> %374) secondary(%375 -> %378) (<0:1:0> -> <0:1:3> : !VPURegMapped.Index<0:1:0> -> !VPURegMapped.Index<0:1:3>) -> <0:0:2> {associated_execution_group_index = 0 : ui64, associated_task_type = #VPURegMapped.task_type<ActKernelRange>, associated_tile_index = 0 : ui64}
  %ft3 = VPURegMapped.FetchTask previousTask(%ft2 : !VPURegMapped.Index<0:0:2>) primary(%177 -> %180) secondary(%181 -> %184) (<0:0:0> -> <0:0:3> : !VPURegMapped.Index<0:0:0> -> !VPURegMapped.Index<0:0:3>) -> <0:0:3> {associated_execution_group_index = 0 : ui64, associated_task_type = #VPURegMapped.task_type<ActKernelRange>, associated_tile_index = 0 : ui64}
  %323 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64} inputs(%2 : memref<0x0x0x0xi32, @DDR>) outputs(%3 : memref<0x0x0x0xi32, @DDR>) previousDMA(%ft3 : !VPURegMapped.Index<0:0:3>) updates(%42 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:4>
  %324 = VPUMI40XX.NNDMA {is_out_of_order, port = 0 : i64} inputs(%0 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) outputs(%22, %23, %345, %346 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>, memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>, memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>, memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) previousDMA(%323 : !VPURegMapped.Index<0:0:4>) waits(%42 : !VPURegMapped.Index<0:0:0>) updates(%43 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>, outputType = !VPUIP.DistributedBuffer<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1000, 1, 1], [1, 1000, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1000, 1, 1], [1, 1000, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>>) -> !VPURegMapped.Index<0:0:5>
  %325 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%21 : memref<1x1000x1x1xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x1000x1x1xf16, @DDR>) waits(%47 : !VPURegMapped.Index<0:0:5>) updates(%48 : !VPURegMapped.Index<0:0:6>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x1000x1x1xf16, [@CMX_NN, 0]>, outputType = memref<1x1000x1x1xf16, @DDR>>) -> !VPURegMapped.Index<0:1:0>
  %326 = VPUMI40XX.PlatformInfo -> <0:0:0>
  %327 = VPUMI40XX.ActShaveRt kernel("nnActEntry") -> !VPURegMapped.Index<0:0:0>
  %328 = VPUMI40XX.MappedInference {workloadManagementBarrierProgrammingMode = #VPURegMapped.workload_management_barrier_programming_mode<NO_BARRIER_DMAS_SCHEDULED>} dmas((%ft0, %325) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) actKernelRanges((%177, %371), (%313, %387) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>), (!VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<1:1:0>)) actKernelInvocations((%181, %375), (%317, %391) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>), (!VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<1:1:0>)) barriers(%42 : !VPURegMapped.Index<0:0:0>) actShaveRt(%327 : !VPURegMapped.Index<0:0:0>) dmaHwpBase(%4 : memref<16xui32, [@CMX_NN, 0]>) dmaCount([[6, 1], [0, 0]]) invariantCount([0, 0]) variantCount([0, 0]) actKernelRangesCount([[4, 4], [4, 4]]) actKernelInvocationsCount([[4, 4], [4, 4]]) mediaCount(0) barrierCount(7) finalBarrierId(6) barrierConfigurationTasksCount(224) -> !VPURegMapped.Index<0:0:0>
  return %arg1 : memref<1x1000x1x1xf16, @DDR>
  }
}

//CHECK:  [[VAL42:%.+]] = VPUMI40XX.ConfigureBarrier

//CHECK:  [[ENQ0:%.+]] = VPURegMapped.Enqueue at([[VAL42]] : !VPURegMapped.Index<0:0:0>)
//CHECK:  [[ENQ1:%.+]] = VPURegMapped.Enqueue previousTaskIdx([[ENQ0]] : !VPURegMapped.Index<0:0:0>) at([[VAL42]] : !VPURegMapped.Index<0:0:0>)
//CHECK:  [[ENQ2:%.+]] = VPURegMapped.Enqueue previousTaskIdx([[ENQ1]] : !VPURegMapped.Index<0:0:1>) at([[VAL42]] : !VPURegMapped.Index<0:0:0>)
//CHECK:  [[ENQ3:%.+]] = VPURegMapped.Enqueue previousTaskIdx([[ENQ2]] : !VPURegMapped.Index<0:0:2>) at([[VAL42]] : !VPURegMapped.Index<0:0:0>)
