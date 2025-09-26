//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --barrier-topological-mapping %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

module @Convolution {
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

  // Schedule:
  //  DMA0DDR_0 -> BAR0 -> DMA0DDR_1 -> BAR1 -> DPU0 -> BAR2 -> DPU1 -> BAR3 -> DMA0CMX_0 -> BAR4
  //                \---> DMA1DDR_0 ------------------>/
  //
  // Expected barrier dependencies:
  //  BAR0 -> BAR1 -> BAR2 -> BAR3 -> BAR4
  //     \------------>/

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
    %bufDdr0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %bufDdr1 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>

    // Reorder the barriers to make sure final order is updated
    %bar0 = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, isStartBarrier, producer_count = 1 : ui8} <4, -1> -> !VPURegMapped.Index<0:0:1>
    %bar1 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <0, -1> -> !VPURegMapped.Index<0:0:3>
    %bar2 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 2 : ui8} <1, -1> -> !VPURegMapped.Index<0:0:2>
    %bar3 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <2, -1> -> !VPURegMapped.Index<0:0:0>
    %bar4 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, isFinalBarrier, producer_count = 1 : ui8} <3, -1> -> !VPURegMapped.Index<0:0:4>

    %Dma0Ddr0 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 0 : i32, srcWidth = 0 : i32, srcStride = 0 : i32, srcPlaneStride = 0 : i32, dstWidth = 0 : i32, dstStride = 0 : i32, dstPlaneStride = 0 : i32>, port = 0 : i64} inputs(%bufDdr0 : memref<1x1x1x1xi32, @DDR>) outputs(%bufDdr1 : memref<1x1x1x1xi32, @DDR>) updates(%bar0 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %Dma0Ddr1 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x16x16x16xf16, @DDR>) outputs(%2 : memref<1x16x16x16xf16, [@CMX_NN, 0]>) previousDMA(%Dma0Ddr0 : !VPURegMapped.Index<0:0:0>) waits(%bar0 : !VPURegMapped.Index<0:0:1>) updates(%bar1 : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %Dma1Ddr0 = VPUMI40XX.NNDMA {is_out_of_order, port = 1 : i64} inputs(%cst : memref<1x1x1x4864xui8>) outputs(%5 : memref<1x1x1x4864xui8, [@CMX_NN, 0]>) waits(%bar0 : !VPURegMapped.Index<0:0:1>) updates(%bar2 : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:0:0>
    %Dma0Cmx0 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x16x14x14xf16, @DDR>) waits(%bar3 : !VPURegMapped.Index<0:0:0>) updates(%bar4 : !VPURegMapped.Index<0:0:4>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %DpuInv0 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64} input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%bar1 : !VPURegMapped.Index<0:0:3>) updates(%bar2 : !VPURegMapped.Index<0:0:2>) -> <0:0:0> PPE : {
    }
    %DpuInv1 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} previousTask(%DpuInv0 : !VPURegMapped.Index<0:0:0>) input(%7 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) waits(%bar2 : !VPURegMapped.Index<0:0:2>) updates(%bar3 : !VPURegMapped.Index<0:0:0>) -> <0:0:1> PPE : {
    }
    %DpuVar0 = VPUMI40XX.DPUVariant calls(%DpuInv0 : <0:0:0>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
    %DpuVar1 = VPUMI40XX.DPUVariant previousTask(%DpuVar0 : !VPURegMapped.Index<0:0:0>) calls(%DpuInv1 : <0:0:1>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
    %mi = VPUMI40XX.MappedInference dmas((%Dma0Ddr0, %Dma0Cmx0), (%Dma1Ddr0) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>), (!VPURegMapped.Index<1:0:0>)) invariants(%DpuInv0 : !VPURegMapped.Index<0:0:0>) variants(%DpuVar0 : !VPURegMapped.Index<0:0:0>) barriers(%bar3 : !VPURegMapped.Index<0:0:0>) dmaCount([[2, 1], [1, 0]]) invariantCount([2]) variantCount([2]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(5) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x14x14xf16, @DDR>
  }
}

// CHECK: [[BAR0:%.+]] = VPUMI40XX.ConfigureBarrier
// CHECK-SAME: isStartBarrier
// CHECK-SAME: <4, -1>

// CHECK: [[BAR1:%.+]] = VPUMI40XX.ConfigureBarrier
// CHECK-SAME: ([[BAR0]] : !VPURegMapped.Index<0:0:0>)
// CHECK-SAME: <0, -1>

// CHECK: [[BAR2:%.+]] = VPUMI40XX.ConfigureBarrier
// CHECK-SAME: ([[BAR0]], [[BAR1]] : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:1>)
// CHECK-SAME: <1, -1>

// CHECK: [[BAR3:%.+]] = VPUMI40XX.ConfigureBarrier
// CHECK-SAME: ([[BAR2]] : !VPURegMapped.Index<0:0:2>)
// CHECK-SAME: <2, -1>

// CHECK: [[BAR4:%.+]] = VPUMI40XX.ConfigureBarrier
// CHECK-SAME: isFinalBarrier
// CHECK-SAME: ([[BAR3]] : !VPURegMapped.Index<0:0:3>)
// CHECK-SAME: <3, -1>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

module @Convolution {
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

  // Schedule:
  //  DMA1DDR_0 -> BAR0 -> DMA0DDR_0 -> BAR1 -> DMA0DDR_1 -> BAR2 -> DMA1DDR_2 -> BAR3
  //                 \---> DMA1DDR_1 ------------------------>/
  //
  // Expected barrier dependencies:
  //  BAR0 -> BAR1 -> BAR2 -> BAR3 -> BAR4
  //     \------------>/

  func.func @main(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<1x16x14x14xf16, @DDR>) -> memref<1x16x14x14xf16, @DDR> {

    %bufDdr = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x16x16xf16, @DDR>
    %bufCmx = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x16x16xf16, [@CMX_NN, 0]>

    // Reorder the barriers to make sure final order is updated
    %bar0 = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 1 : ui8} <0, -1> -> !VPURegMapped.Index<0:0:0>
    %bar1 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, isStartBarrier, producer_count = 1 : ui8} <1, -1> -> !VPURegMapped.Index<0:0:2>
    %bar2 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 2 : ui8} <2, -1> -> !VPURegMapped.Index<0:0:1>
    %bar3 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, isFinalBarrier, producer_count = 1 : ui8} <3, -1> -> !VPURegMapped.Index<0:0:3>

    %Dma1Ddr0 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%bufDdr : memref<1x16x16x16xf16, @DDR>) outputs(%bufCmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) updates(%bar0 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:0:0>
    %Dma0Ddr0 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%bufDdr : memref<1x16x16x16xf16, @DDR>) outputs(%bufCmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) waits(%bar0 : !VPURegMapped.Index<0:0:0>) updates(%bar1 : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %Dma1Ddr1 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%bufDdr : memref<1x16x16x16xf16, @DDR>) outputs(%bufCmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) previousDMA(%Dma1Ddr0 : !VPURegMapped.Index<1:0:0>) waits(%bar0 : !VPURegMapped.Index<0:0:0>) updates(%bar2 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:0:1>
    %Dma0Ddr1 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%bufDdr : memref<1x16x16x16xf16, @DDR>) outputs(%bufCmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) previousDMA(%Dma0Ddr0 : !VPURegMapped.Index<0:0:0>) waits(%bar1 : !VPURegMapped.Index<0:0:2>) updates(%bar2 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %Dma1Ddr2 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%bufDdr : memref<1x16x16x16xf16, @DDR>) outputs(%bufCmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) previousDMA(%Dma1Ddr1 : !VPURegMapped.Index<1:0:1>) waits(%bar2 : !VPURegMapped.Index<0:0:1>) updates(%bar3 : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:0:2>


    %mi = VPUMI40XX.MappedInference dmas((%Dma0Ddr0), (%Dma1Ddr0) : (!VPURegMapped.Index<0:0:0>), (!VPURegMapped.Index<1:0:0>)) barriers(%bar0 : !VPURegMapped.Index<0:0:0>) dmaCount([[2, 0], [3, 0]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(4) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x14x14xf16, @DDR>
  }
}

// CHECK: [[BAR0:%.+]] = VPUMI40XX.ConfigureBarrier
// CHECK-SAME: <0, -1>

// CHECK: [[BAR1:%.+]] = VPUMI40XX.ConfigureBarrier
// CHECK-SAME: isStartBarrier
// CHECK-SAME: ([[BAR0]] : !VPURegMapped.Index<0:0:0>)
// CHECK-SAME: <1, -1>

// CHECK: [[BAR2:%.+]] = VPUMI40XX.ConfigureBarrier
// CHECK-SAME: ([[BAR1]], [[BAR0]] : !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:0>)
// CHECK-SAME: <2, -1>

// CHECK: [[BAR3:%.+]] = VPUMI40XX.ConfigureBarrier
// CHECK-SAME: isFinalBarrier
// CHECK-SAME: ([[BAR2]] : !VPURegMapped.Index<0:0:2>)
// CHECK-SAME: <3, -1>
