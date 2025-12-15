//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --convert-fetch-dmas-to-fetch-task-ops %s | FileCheck %s
// REQUIRES: arch-NPU50XX

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
    %startIndexes:2, %endIndexes:2 = "VPURegMapped.ExecutionGroup"(%11, %13) ({
      %22 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64} input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%12 : !VPURegMapped.Index<0:0:2>) -> <0:0:0> PPE : {
        VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
      }
      %23 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} previousTask(%22 : !VPURegMapped.Index<0:0:0>) input(%7 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) waits(%12 : !VPURegMapped.Index<0:0:2>) updates(%13 : !VPURegMapped.Index<0:0:3>) -> <0:0:1> PPE : {
        VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
      }
      %24 = VPUMI40XX.DPUVariant calls(%22 : <0:0:0>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
      %25 = VPUMI40XX.DPUVariant previousTask(%24 : !VPURegMapped.Index<0:0:0>) calls(%23 : <0:0:1>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
      "VPURegMapped.GroupYield"(%22, %24, %23, %25) {operandSegmentSizes = array<i32: 2, 2>} : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>) -> ()
    }) {operandSegmentSizes = array<i32: 0, 1, 1>, resultSegmentSizes = array<i32: 2, 2>, task_type = #VPURegMapped.task_type<DPUInvariant>} : (!VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:3>) -> (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>)
    %15 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %16 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %161 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 0 : i32, srcWidth = 0 : i32, srcStride = 0 : i32, srcPlaneStride = 0 : i32, dstWidth = 0 : i32, dstStride = 0 : i32, dstPlaneStride = 0 : i32>, fetch_dma = #VPUIP.FetchDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, group = 0 : i64>, port = 0 : i64, wlmPage = -1 : i64} inputs(%15 : memref<0x0x0x0xi32, @DDR>) outputs(%16 : memref<0x0x0x0xi32, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %17 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 0 : i32, srcWidth = 0 : i32, srcStride = 0 : i32, srcPlaneStride = 0 : i32, dstWidth = 0 : i32, dstStride = 0 : i32, dstPlaneStride = 0 : i32>, port = 1 : i64} inputs(%15 : memref<0x0x0x0xi32, @DDR>) outputs(%16 : memref<0x0x0x0xi32, @DDR>) previousDMA(%161 : !VPURegMapped.Index<0:0:0>) updates(%10 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %18 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%0 : memref<1x16x16x16xf16, @DDR>) outputs(%2 : memref<1x16x16x16xf16, [@CMX_NN, 0]>) previousDMA(%17 : !VPURegMapped.Index<0:0:1>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%11 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %19 = VPUMI40XX.NNDMA {is_out_of_order, port = 1 : i64} inputs(%cst : memref<1x1x1x4864xui8>) outputs(%5 : memref<1x1x1x4864xui8, [@CMX_NN, 0]>) previousDMA(%18 : !VPURegMapped.Index<0:0:2>) updates(%12 : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>
    %20 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x16x14x14xf16, @DDR>) waits(%13 : !VPURegMapped.Index<0:0:3>) updates(%14 : !VPURegMapped.Index<0:0:4>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %21 = VPUMI40XX.MappedInference dmas((%161, %20) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) invariants(%startIndexes#0 : !VPURegMapped.Index<0:0:0>) variants(%startIndexes#1 : !VPURegMapped.Index<0:0:0>) barriers(%10 : !VPURegMapped.Index<0:0:0>) dmaCount([[3, 1]]) invariantCount([2]) variantCount([2]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(5) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x14x14xf16, @DDR>
  }
}


// CHECK: [[VAL10:%.*]] = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <4, -1> -> !VPURegMapped.Index<0:0:0>
// CHECK: [[VAL11:%.*]] = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}(%10 : !VPURegMapped.Index<0:0:0>) <0, -1> -> !VPURegMapped.Index<0:0:1>
// CHECK: [[VAL12:%.*]] = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 2 : ui8}(%11 : !VPURegMapped.Index<0:0:1>) <1, -1> -> !VPURegMapped.Index<0:0:2>
// CHECK: [[VAL13:%.*]] = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}(%12 : !VPURegMapped.Index<0:0:2>) <2, -1> -> !VPURegMapped.Index<0:0:3>
// CHECK: [[VAL14:%.*]] = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, isFinalBarrier, producer_count = 1 : ui8}(%13 : !VPURegMapped.Index<0:0:3>) <3, -1> -> !VPURegMapped.Index<0:0:4>
// CHECK: [[startIndex:%.+]], [[startIndex:%.+]] = "VPURegMapped.ExecutionGroup"([[VAL11]], [[VAL13]])
// CHECK: [[VAL23:%.*]] = VPUMI40XX.DPUInvariant
// CHECK: [[VAL24:%.*]] = VPUMI40XX.DPUInvariant
// CHECK: [[VAL25:%.*]] = VPUMI40XX.DPUVariant
// CHECK: [[VAL26:%.*]] = VPUMI40XX.DPUVariant
// CHECK: "VPURegMapped.GroupYield"([[VAL23]], [[VAL25]], [[VAL24]], [[VAL26]]) <{operandSegmentSizes = array<i32: 2, 2>}> : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>) -> ()
// CHECK: VPURegMapped.FetchTask primary(%startIndexes#0 -> %endIndexes#0) secondary(%startIndexes#1 -> %endIndexes#1)

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
  %71 = VPURT.DeclareBuffer <CMX_NN> [0] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %72 = VPURT.DeclareBuffer <CMX_NN> [0] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %73 = VPURT.DeclareBuffer <CMX_NN> [0] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %74 = VPURT.DeclareBuffer <CMX_NN> [0] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %75 = VPURT.DeclareBuffer <CMX_NN> [0] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %76 = VPURT.DeclareBuffer <CMX_NN> [0] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %77 = VPURT.DeclareBuffer <CMX_NN> [0] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %78 = VPURT.DeclareBuffer <CMX_NN> [0] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  // buffers for KernelParams of ActKernelInvocation ops in tile 1, list 1:
  %79 = VPURT.DeclareBuffer <CMX_NN> [1] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %80 = VPURT.DeclareBuffer <CMX_NN> [1] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %81 = VPURT.DeclareBuffer <CMX_NN> [1] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %82 = VPURT.DeclareBuffer <CMX_NN> [1] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %83 = VPURT.DeclareBuffer <CMX_NN> [1] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %84 = VPURT.DeclareBuffer <CMX_NN> [1] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %85 = VPURT.DeclareBuffer <CMX_NN> [1] <6048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>
  %86 = VPURT.DeclareBuffer <CMX_NN> [1] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>

  %87 = VPURT.DeclareBuffer <CMX_NN> [0] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %88 = VPURT.DeclareBuffer <CMX_NN> [1] <4048> -> memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>

  %24 = VPUMI40XX.DeclareKernelText kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
  %25 = VPUMI40XX.DeclareKernelEntry kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
  %26 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
  %89 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:1:0>
  %27 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:0:0>
  %90 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:1:0>
  %28 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:1>
  %91 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:1:1>
  %29 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:0:1>
  %92 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:1:1>
  %30 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:2>
  %93 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:1:2>
  %31 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:0:2>
  %94 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:1:2>
  %32 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:3>
  %95 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:1:3>
  %33 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:0:3>
  %96 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:1:3>
  %34 = VPUMI40XX.KernelParams inputs(%5 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%9 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:0:0>
  %97 = VPUMI40XX.KernelParams inputs(%71 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%72 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:1:0>
  %35 = VPUMI40XX.KernelParams inputs(%6 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%10 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:0:0>
  %98 = VPUMI40XX.KernelParams inputs(%79 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%80 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:1:0>
  %36 = VPUMI40XX.KernelParams inputs(%7 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%13 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:0:1>
  %99 = VPUMI40XX.KernelParams inputs(%73 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%74 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:1:1>
  %37 = VPUMI40XX.KernelParams inputs(%8 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%14 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:0:1>
  %100 = VPUMI40XX.KernelParams inputs(%81 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%82 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:1:1>
  %38 = VPUMI40XX.KernelParams inputs(%11 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%17 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:0:2>
  %101 = VPUMI40XX.KernelParams inputs(%75 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%76 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:1:2>
  %39 = VPUMI40XX.KernelParams inputs(%12 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%18 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:0:2>
  %102 = VPUMI40XX.KernelParams inputs(%83 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%84 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:1:2>
  %40 = VPUMI40XX.KernelParams inputs(%15 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%19 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:0:3>
  %103 = VPUMI40XX.KernelParams inputs(%77 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%78 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:1:3>
  %41 = VPUMI40XX.KernelParams inputs(%16 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%20 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:0:3>
  %104 = VPUMI40XX.KernelParams inputs(%85 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) outputs(%86 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:1:3>
  %42 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, isStartBarrier, producer_count = 1 : ui8} <0, -1> -> !VPURegMapped.Index<0:0:0>
  %43 = VPUMI40XX.ConfigureBarrier {consumer_count = 4 : ui8, producer_count = 1 : ui8}(%42 : !VPURegMapped.Index<0:0:0>) <1, -1> -> !VPURegMapped.Index<0:0:1>
  %44 = VPUMI40XX.ConfigureBarrier {consumer_count = 4 : ui8, producer_count = 4 : ui8}(%43 : !VPURegMapped.Index<0:0:1>) <2, -1> -> !VPURegMapped.Index<0:0:2>
  %45 = VPUMI40XX.ConfigureBarrier {consumer_count = 4 : ui8, producer_count = 4 : ui8}(%44 : !VPURegMapped.Index<0:0:2>) <3, -1> -> !VPURegMapped.Index<0:0:3>
  %46 = VPUMI40XX.ConfigureBarrier {consumer_count = 4 : ui8, producer_count = 4 : ui8}(%45 : !VPURegMapped.Index<0:0:3>) <4, -1> -> !VPURegMapped.Index<0:0:4>
  %47 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 4 : ui8}(%46 : !VPURegMapped.Index<0:0:4>) <5, -1> -> !VPURegMapped.Index<0:0:5>
  %48 = VPUMI40XX.ConfigureBarrier {consumer_count = 0 : ui8, isFinalBarrier, producer_count = 1 : ui8}(%47 : !VPURegMapped.Index<0:0:5>) <6, -1> -> !VPURegMapped.Index<0:0:6>
  %startIndexes:2, %endIndexes:2 = "VPURegMapped.ExecutionGroup"(%43, %47) <{operandSegmentSizes = array<i32: 0, 1, 1>, resultSegmentSizes = array<i32: 2, 2>, task_type = #VPURegMapped.task_type<ActKernelRange>}> ({
    %55 = VPUMI40XX.ActKernelRange kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%26 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
    %56 = VPUMI40XX.ActKernelRange previousTask(%55 : !VPURegMapped.Index<0:0:0>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%28 : !VPURegMapped.Index<0:0:1>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:1>
    %57 = VPUMI40XX.ActKernelRange previousTask(%56 : !VPURegMapped.Index<0:0:1>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%30 : !VPURegMapped.Index<0:0:2>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:2>
    %58 = VPUMI40XX.ActKernelRange previousTask(%57 : !VPURegMapped.Index<0:0:2>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%32 : !VPURegMapped.Index<0:0:3>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:3>
    %59 = VPUMI40XX.ActKernelInvocation range_index(%55 : <0:0:0>) kernel_params(%34 : <0:0:0>) waits(%43 : !VPURegMapped.Index<0:0:1>) updates(%44 : !VPURegMapped.Index<0:0:2>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %60 = VPUMI40XX.ActKernelInvocation previousTask(%59 : !VPURegMapped.Index<0:0:0>) range_index(%56 : <0:0:1>) kernel_params(%36 : <0:0:1>) waits(%44 : !VPURegMapped.Index<0:0:2>) updates(%45 : !VPURegMapped.Index<0:0:3>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
    %61 = VPUMI40XX.ActKernelInvocation previousTask(%60 : !VPURegMapped.Index<0:0:1>) range_index(%57 : <0:0:2>) kernel_params(%38 : <0:0:2>) waits(%45 : !VPURegMapped.Index<0:0:3>) updates(%46 : !VPURegMapped.Index<0:0:4>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>
    %62 = VPUMI40XX.ActKernelInvocation {lastSecondaryTaskInExecutionGroup} previousTask(%61 : !VPURegMapped.Index<0:0:2>) range_index(%58 : <0:0:3>) kernel_params(%40 : <0:0:3>) waits(%46 : !VPURegMapped.Index<0:0:4>) updates(%47 : !VPURegMapped.Index<0:0:5>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:3>
    "VPURegMapped.GroupYield"(%55, %59, %58, %62) <{operandSegmentSizes = array<i32: 2, 2>}> : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:3>, !VPURegMapped.Index<0:0:3>) -> ()
  }) : (!VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:5>) -> (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:3>, !VPURegMapped.Index<0:0:3>)
  %startIndexes_0:2, %endIndexes_1:2 = "VPURegMapped.ExecutionGroup"(%43, %47) <{operandSegmentSizes = array<i32: 0, 1, 1>, resultSegmentSizes = array<i32: 2, 2>, task_type = #VPURegMapped.task_type<ActKernelRange>}> ({
    %55 = VPUMI40XX.ActKernelRange kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%89 : !VPURegMapped.Index<0:1:0>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:1:0>
    %56 = VPUMI40XX.ActKernelRange previousTask(%55 : !VPURegMapped.Index<0:1:0>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%91 : !VPURegMapped.Index<0:1:1>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:1:1>
    %57 = VPUMI40XX.ActKernelRange previousTask(%56 : !VPURegMapped.Index<0:1:1>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%93 : !VPURegMapped.Index<0:1:2>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:1:2>
    %58 = VPUMI40XX.ActKernelRange previousTask(%57 : !VPURegMapped.Index<0:1:2>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%95 : !VPURegMapped.Index<0:1:3>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:1:3>
    %59 = VPUMI40XX.ActKernelInvocation range_index(%55 : <0:1:0>) kernel_params(%97 : <0:1:0>) waits(%43 : !VPURegMapped.Index<0:0:1>) updates(%44 : !VPURegMapped.Index<0:0:2>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:1:0>
    %60 = VPUMI40XX.ActKernelInvocation previousTask(%59 : !VPURegMapped.Index<0:1:0>) range_index(%56 : <0:1:1>) kernel_params(%99 : <0:1:1>) waits(%44 : !VPURegMapped.Index<0:0:2>) updates(%45 : !VPURegMapped.Index<0:0:3>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:1:1>
    %61 = VPUMI40XX.ActKernelInvocation previousTask(%60 : !VPURegMapped.Index<0:1:1>) range_index(%57 : <0:1:2>) kernel_params(%101 : <0:1:2>) waits(%45 : !VPURegMapped.Index<0:0:3>) updates(%46 : !VPURegMapped.Index<0:0:4>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:1:2>
    %62 = VPUMI40XX.ActKernelInvocation {lastSecondaryTaskInExecutionGroup} previousTask(%61 : !VPURegMapped.Index<0:1:2>) range_index(%58 : <0:1:3>) kernel_params(%103 : <0:1:3>) waits(%46 : !VPURegMapped.Index<0:0:4>) updates(%47 : !VPURegMapped.Index<0:0:5>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:1:3>
    "VPURegMapped.GroupYield"(%55, %59, %58, %62) <{operandSegmentSizes = array<i32: 2, 2>}> : (!VPURegMapped.Index<0:1:0>, !VPURegMapped.Index<0:1:0>, !VPURegMapped.Index<0:1:3>, !VPURegMapped.Index<0:1:3>) -> ()
  }) : (!VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:5>) -> (!VPURegMapped.Index<0:1:0>, !VPURegMapped.Index<0:1:0>, !VPURegMapped.Index<0:1:3>, !VPURegMapped.Index<0:1:3>)
  %startIndexes_2:2, %endIndexes_3:2 = "VPURegMapped.ExecutionGroup"(%43, %47) <{operandSegmentSizes = array<i32: 0, 1, 1>, resultSegmentSizes = array<i32: 2, 2>, task_type = #VPURegMapped.task_type<ActKernelRange>}> ({
    %55 = VPUMI40XX.ActKernelRange kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%27 : !VPURegMapped.Index<1:0:0>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:0:0>
    %56 = VPUMI40XX.ActKernelRange previousTask(%55 : !VPURegMapped.Index<1:0:0>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%29 : !VPURegMapped.Index<1:0:1>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:0:1>
    %57 = VPUMI40XX.ActKernelRange previousTask(%56 : !VPURegMapped.Index<1:0:1>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%31 : !VPURegMapped.Index<1:0:2>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:0:2>
    %58 = VPUMI40XX.ActKernelRange previousTask(%57 : !VPURegMapped.Index<1:0:2>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%33 : !VPURegMapped.Index<1:0:3>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:0:3>
    %59 = VPUMI40XX.ActKernelInvocation range_index(%55 : <1:0:0>) kernel_params(%35 : <1:0:0>) waits(%43 : !VPURegMapped.Index<0:0:1>) updates(%44 : !VPURegMapped.Index<0:0:2>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:0>
    %60 = VPUMI40XX.ActKernelInvocation previousTask(%59 : !VPURegMapped.Index<1:0:0>) range_index(%56 : <1:0:1>) kernel_params(%37 : <1:0:1>) waits(%44 : !VPURegMapped.Index<0:0:2>) updates(%45 : !VPURegMapped.Index<0:0:3>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:1>
    %61 = VPUMI40XX.ActKernelInvocation previousTask(%60 : !VPURegMapped.Index<1:0:1>) range_index(%57 : <1:0:2>) kernel_params(%39 : <1:0:2>) waits(%45 : !VPURegMapped.Index<0:0:3>) updates(%46 : !VPURegMapped.Index<0:0:4>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:2>
    %62 = VPUMI40XX.ActKernelInvocation {lastSecondaryTaskInExecutionGroup} previousTask(%61 : !VPURegMapped.Index<1:0:2>) range_index(%58 : <1:0:3>) kernel_params(%41 : <1:0:3>) waits(%46 : !VPURegMapped.Index<0:0:4>) updates(%47 : !VPURegMapped.Index<0:0:5>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:3>
    "VPURegMapped.GroupYield"(%55, %59, %58, %62) <{operandSegmentSizes = array<i32: 2, 2>}> : (!VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<1:0:3>, !VPURegMapped.Index<1:0:3>) -> ()
  }) : (!VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:5>) -> (!VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<1:0:3>, !VPURegMapped.Index<1:0:3>)
  %startIndexes_4:2, %endIndexes_5:2 = "VPURegMapped.ExecutionGroup"(%43, %47) <{operandSegmentSizes = array<i32: 0, 1, 1>, resultSegmentSizes = array<i32: 2, 2>, task_type = #VPURegMapped.task_type<ActKernelRange>}> ({
    %55 = VPUMI40XX.ActKernelRange kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%90 : !VPURegMapped.Index<1:1:0>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:1:0>
    %56 = VPUMI40XX.ActKernelRange previousTask(%55 : !VPURegMapped.Index<1:1:0>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%92 : !VPURegMapped.Index<1:1:1>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:1:1>
    %57 = VPUMI40XX.ActKernelRange previousTask(%56 : !VPURegMapped.Index<1:1:1>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%94 : !VPURegMapped.Index<1:1:2>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:1:2>
    %58 = VPUMI40XX.ActKernelRange previousTask(%57 : !VPURegMapped.Index<1:1:2>) kernel_text_index(%24 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%96 : !VPURegMapped.Index<1:1:3>) kernel_entry_index(%25 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:1:3>
    %59 = VPUMI40XX.ActKernelInvocation range_index(%55 : <1:1:0>) kernel_params(%98 : <1:1:0>) waits(%43 : !VPURegMapped.Index<0:0:1>) updates(%44 : !VPURegMapped.Index<0:0:2>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:1:0>
    %60 = VPUMI40XX.ActKernelInvocation previousTask(%59 : !VPURegMapped.Index<1:1:0>) range_index(%56 : <1:1:1>) kernel_params(%100 : <1:1:1>) waits(%44 : !VPURegMapped.Index<0:0:2>) updates(%45 : !VPURegMapped.Index<0:0:3>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:1:1>
    %61 = VPUMI40XX.ActKernelInvocation previousTask(%60 : !VPURegMapped.Index<1:1:1>) range_index(%57 : <1:1:2>) kernel_params(%102 : <1:1:2>) waits(%45 : !VPURegMapped.Index<0:0:3>) updates(%46 : !VPURegMapped.Index<0:0:4>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:1:2>
    %62 = VPUMI40XX.ActKernelInvocation {lastSecondaryTaskInExecutionGroup} previousTask(%61 : !VPURegMapped.Index<1:1:2>) range_index(%58 : <1:1:3>) kernel_params(%104 : <1:1:3>) waits(%46 : !VPURegMapped.Index<0:0:4>) updates(%47 : !VPURegMapped.Index<0:0:5>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:1:3>
    "VPURegMapped.GroupYield"(%55, %59, %58, %62) <{operandSegmentSizes = array<i32: 2, 2>}> : (!VPURegMapped.Index<1:1:0>, !VPURegMapped.Index<1:1:0>, !VPURegMapped.Index<1:1:3>, !VPURegMapped.Index<1:1:3>) -> ()
  }) : (!VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:5>) -> (!VPURegMapped.Index<1:1:0>, !VPURegMapped.Index<1:1:0>, !VPURegMapped.Index<1:1:3>, !VPURegMapped.Index<1:1:3>)

  %491 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64, fetch_dma = #VPUIP.FetchDMAAttr<<SHAVE_ACT>, tile = 1 : i64, list = 1 : i64, group = 0 : i64>, wlmPage = -1 : i64} inputs(%2 : memref<0x0x0x0xi32, @DDR>) outputs(%3 : memref<0x0x0x0xi32, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

  %492 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, fetch_dma = #VPUIP.FetchDMAAttr<<SHAVE_ACT>, tile = 1 : i64, list = 0 : i64, group = 0 : i64>, port = 0 : i64, wlmPage = -1 : i64} inputs(%2 : memref<0x0x0x0xi32, @DDR>) outputs(%3 : memref<0x0x0x0xi32, @DDR>) previousDMA(%491 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>

  %493 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64, fetch_dma = #VPUIP.FetchDMAAttr<<SHAVE_ACT>, tile = 0 : i64, list = 1 : i64, group = 0 : i64>, wlmPage = -1 : i64} inputs(%2 : memref<0x0x0x0xi32, @DDR>) outputs(%3 : memref<0x0x0x0xi32, @DDR>) previousDMA(%492 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>

   %494 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64, fetch_dma = #VPUIP.FetchDMAAttr<<SHAVE_ACT>, tile = 0 : i64, list = 0 : i64, group = 0 : i64>, wlmPage = -1 : i64} inputs(%2 : memref<0x0x0x0xi32, @DDR>) outputs(%3 : memref<0x0x0x0xi32, @DDR>) previousDMA(%493 : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>

  %49 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64} inputs(%2 : memref<0x0x0x0xi32, @DDR>) outputs(%3 : memref<0x0x0x0xi32, @DDR>) previousDMA(%494 : !VPURegMapped.Index<0:0:3>) updates(%42 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:4>

  %50 = VPUMI40XX.NNDMA {is_out_of_order, port = 0 : i64} inputs(%0 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) outputs(%22, %23, %87, %88 : memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>, memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>, memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>, memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 1]>) previousDMA(%49 : !VPURegMapped.Index<0:0:4>) waits(%42 : !VPURegMapped.Index<0:0:0>) updates(%43 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>, outputType = !VPUIP.DistributedBuffer<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1000, 1, 1], [1, 1000, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1000, 1, 1], [1, 1000, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>>) -> !VPURegMapped.Index<0:0:5>

  %51 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%21 : memref<1x1000x1x1xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x1000x1x1xf16, @DDR>) waits(%47 : !VPURegMapped.Index<0:0:5>) updates(%48 : !VPURegMapped.Index<0:0:6>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x1000x1x1xf16, [@CMX_NN, 0]>, outputType = memref<1x1000x1x1xf16, @DDR>>) -> !VPURegMapped.Index<0:1:0>
  %52 = VPUMI40XX.PlatformInfo -> <0:0:0>
  %53 = VPUMI40XX.ActShaveRt kernel("nnActEntry") -> !VPURegMapped.Index<0:0:0>
  %54 = VPUMI40XX.MappedInference dmas((%491, %51) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) actKernelRanges((%startIndexes#0, %startIndexes_0#0), (%startIndexes_2#0, %startIndexes_4#0) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>), (!VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<1:1:0>)) actKernelInvocations((%startIndexes#1, %startIndexes_0#1), (%startIndexes_2#1, %startIndexes_4#1) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>), (!VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<1:1:0>)) barriers(%42 : !VPURegMapped.Index<0:0:0>) actShaveRt(%53 : !VPURegMapped.Index<0:0:0>) dmaHwpBase(%4 : memref<16xui32, [@CMX_NN, 0]>) dmaCount([[3, 1], [0, 0]]) invariantCount([0, 0]) variantCount([0, 0]) actKernelRangesCount([[4, 4], [4, 4]]) actKernelInvocationsCount([[4, 4], [4, 4]]) mediaCount(0) barrierCount(7) -> !VPURegMapped.Index<0:0:0>
  return %arg1 : memref<1x1000x1x1xf16, @DDR>
  }
}

//CHECK:  [[VAL42:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK:  [[VAL43:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK:  [[VAL44:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK:  [[VAL45:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK:  [[VAL46:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK:  [[VAL47:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK:  [[VAL48:%.+]] = VPUMI40XX.ConfigureBarrier
//CHECK-NEXT: %[[START:[a-zA-Z_][a-zA-Z0-9_]*]]:[[NUM1:[0-9]+]], %[[END:[a-zA-Z_][a-zA-Z0-9_]*]]:[[NUM2:[0-9]+]] = "VPURegMapped.ExecutionGroup"([[VAL43]], [[VAL47]])
//CHECK: %[[START1:[a-zA-Z_][a-zA-Z0-9_]*]]:[[IGNORE:[0-9]+]], %[[END1:[a-zA-Z_][a-zA-Z0-9_]*]]:[[IGNORE1:[0-9]+]] = "VPURegMapped.ExecutionGroup"([[VAL43]], [[VAL47]])
//CHECK: %[[START2:[a-zA-Z_][a-zA-Z0-9_]*]]:[[IGNORE2:[0-9]+]], %[[END2:[a-zA-Z_][a-zA-Z0-9_]*]]:[[IGNORE3:[0-9]+]] = "VPURegMapped.ExecutionGroup"([[VAL43]], [[VAL47]])
//CHECK: %[[START3:[a-zA-Z_][a-zA-Z0-9_]*]]:[[IGNORE4:[0-9]+]], %[[END3:[a-zA-Z_][a-zA-Z0-9_]*]]:[[IGNORE5:[0-9]+]] = "VPURegMapped.ExecutionGroup"([[VAL43]], [[VAL47]])
//CHECK:  [[VAL49:%.+]] = VPURegMapped.FetchTask primary(%[[START3]]#0 -> %[[END3]]#0) secondary(%[[START3]]#1 -> %[[END3]]#1) (<1:1:0> -> <1:1:3> : !VPURegMapped.Index<1:1:0> -> !VPURegMapped.Index<1:1:3>) -> <0:0:0>
//CHECK:  [[VAL50:%.+]] = VPURegMapped.FetchTask previousTask([[VAL49]] : !VPURegMapped.Index<0:0:0>) primary(%[[START2]]#0 -> %[[END2]]#0) secondary(%[[START2]]#1 -> %[[END2]]#1) (<1:0:0> -> <1:0:3> : !VPURegMapped.Index<1:0:0> -> !VPURegMapped.Index<1:0:3>) -> <0:0:1>
//CHECK:  [[VAL51:%.+]] = VPURegMapped.FetchTask previousTask([[VAL50]] : !VPURegMapped.Index<0:0:1>) primary(%[[START1]]#0 -> %[[END1]]#0) secondary(%[[START1]]#1 -> %[[END1]]#1) (<0:1:0> -> <0:1:3> : !VPURegMapped.Index<0:1:0> -> !VPURegMapped.Index<0:1:3>) -> <0:0:2>
//CHECK:  [[VAL52:%.+]] = VPURegMapped.FetchTask previousTask([[VAL51]] : !VPURegMapped.Index<0:0:2>) primary(%[[START]]#0 -> %[[END]]#0) secondary(%[[START]]#1 -> %[[END]]#1) (<0:0:0> -> <0:0:3> : !VPURegMapped.Index<0:0:0> -> !VPURegMapped.Index<0:0:3>) -> <0:0:3>
