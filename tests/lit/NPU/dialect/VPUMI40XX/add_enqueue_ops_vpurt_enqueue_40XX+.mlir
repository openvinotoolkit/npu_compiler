//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --add-enqueue-ops="enable-wlm-vpurt-enqueue=true" %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @Convolution attributes {VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.TileResource 1 of @NCE at 1.700000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        IE.MemoryResource 512 bytes of @CMX_NN offset 0
      }
    }
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 1 of @DMA_NN
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork entryPoint : @main inputsInfo : {
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
    %15 = VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
    %16 = VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
    %17 = VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
    %18 = VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
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
    %30 = VPUMI40XX.MappedInference dmas((%25, %29) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) invariants(%19 : !VPURegMapped.Index<0:0:0>) variants(%21 : !VPURegMapped.Index<0:0:0>) barriers(%10 : !VPURegMapped.Index<0:0:0>) dmaCount([[4, 1]]) invariantCount([2]) variantCount([2]) actKernelRangesCount([0]) actKernelInvocationsCount([0]) mediaCount(0) barrierCount(5) -> !VPURegMapped.Index<0:0:0>
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
module @TwoDmaFifosEnqueueOpsForSameBarrierNotNextToEachOther attributes {VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.TileResource 2 of @NCE at 1.700000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        IE.MemoryResource 512 bytes of @CMX_NN offset 0
      }
    }
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 1 of @DMA_NN
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork entryPoint : @main inputsInfo : {
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

    %30 = VPUMI40XX.MappedInference dmas((%dma_ddr_0, %dma_cmx_0) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) barriers(%bar0 : !VPURegMapped.Index<0:0:0>) dmaCount([[4, 2]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([0]) actKernelInvocationsCount([0]) mediaCount(0) barrierCount(7) -> !VPURegMapped.Index<0:0:0>
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
