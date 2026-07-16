//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --convert-fetch-dmas-to-fetch-task-ops %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @Convolution attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
  config.Resources 1 of @NCE at 1.700000e+03 MHz {
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
    DataInfo "output" : tensor<1x16x14x14xf16>
  }
  func.func @main(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<1x16x14x14xf16, @DDR>) -> memref<1x16x14x14xf16, @DDR> {
    %cst = const.Declare memref<1x1x1x4864xui8> = dense<1> : tensor<1x1x1x4864xui8>
    %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x16x16xf16, @DDR>
    %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x14x14xf16, @DDR>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, {order = #NWCH}, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x14x14xf16, [@CMX_NN, 0]>
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <16896> -> memref<1x1x1x4864xui8, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <16896> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <17152> -> memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %10 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}> <4, -1> -> !VPURegMapped.Index<0:0:0>
    %11 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}>(%10 : !VPURegMapped.Index<0:0:0>) <0, -1> -> !VPURegMapped.Index<0:0:1>
    %12 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 2 : ui8}>(%11 : !VPURegMapped.Index<0:0:1>) <1, -1> -> !VPURegMapped.Index<0:0:2>
    %13 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}>(%12 : !VPURegMapped.Index<0:0:2>) <2, -1> -> !VPURegMapped.Index<0:0:3>
    %14 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, isFinalBarrier, producer_count = 1 : ui8}>(%13 : !VPURegMapped.Index<0:0:3>) <3, -1> -> !VPURegMapped.Index<0:0:4>
    %startIndexes:2, %endIndexes:2 = "VPURegMapped.ExecutionGroup"(%11, %13) ({
      %22 = VPUMI40XX.DPUInvariant <{clean_after = 0 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64}>
      input(%6 : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, {order = #NWCH}, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>)
      updates(%12 : !VPURegMapped.Index<0:0:2>) -> <0:0:0> PPE : {
        VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
      }
      %23 = VPUMI40XX.DPUInvariant <{clean_after = 0 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3],
      kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64}>
      previousTask(%22 : !VPURegMapped.Index<0:0:0>) input(%7 : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%9 : memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
      weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) waits(%12 : !VPURegMapped.Index<0:0:2>) updates(%13 : !VPURegMapped.Index<0:0:3>) -> <0:0:1> PPE : {
        VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
      }
      %24 = VPUMI40XX.DPUVariant calls(%22 : <0:0:0>) weights(%6 : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}> -> <0:0:0>
      %25 = VPUMI40XX.DPUVariant previousTask(%24 : !VPURegMapped.Index<0:0:0>) calls(%23 : <0:0:1>) weights(%9 : memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) <{end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}> -> <0:0:1>
      "VPURegMapped.GroupYield"(%22, %24, %23, %25) {operandSegmentSizes = array<i32: 2, 2>} : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>) -> ()
    }) {operandSegmentSizes = array<i32: 0, 1, 1>, resultSegmentSizes = array<i32: 2, 2>, task_type = #VPURegMapped.task_type<DPUInvariant>} : (!VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:3>) -> (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>)
    %15 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %16 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %161 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 0 : i32, srcWidth = 0 : i32, srcStride = 0 : i32, srcPlaneStride = 0 : i32, dstWidth = 0 : i32, dstStride = 0 : i32, dstPlaneStride = 0 : i32>, fetch_dma = #VPUIP.FetchDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, fetchType = <DescriptorGroup>, group = 0 : i64>, port = 0 : i64, wlmPage = -1 : i64}> inputs(%15 : memref<0x0x0x0xi32, @DDR>) outputs(%16 : memref<0x0x0x0xi32, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %17 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 0 : i32, srcWidth = 0 : i32, srcStride = 0 : i32, srcPlaneStride = 0 : i32, dstWidth = 0 : i32, dstStride = 0 : i32, dstPlaneStride = 0 : i32>, port = 1 : i64}> inputs(%15 : memref<0x0x0x0xi32, @DDR>) outputs(%16 : memref<0x0x0x0xi32, @DDR>) previousDMA(%161 : !VPURegMapped.Index<0:0:0>) updates(%10 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %18 = VPUMI40XX.NNDMA <{port = 1 : i64}> inputs(%0 : memref<1x16x16x16xf16, @DDR>) outputs(%2 : memref<1x16x16x16xf16, [@CMX_NN, 0]>) previousDMA(%17 : !VPURegMapped.Index<0:0:1>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%11 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %19 = VPUMI40XX.NNDMA <{is_out_of_order, port = 1 : i64}> inputs(%cst : memref<1x1x1x4864xui8>) outputs(%5 : memref<1x1x1x4864xui8, [@CMX_NN, 0]>) previousDMA(%18 : !VPURegMapped.Index<0:0:2>) updates(%12 : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>
    %20 = VPUMI40XX.NNDMA <{port = 1 : i64}> inputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x16x14x14xf16, @DDR>) waits(%13 : !VPURegMapped.Index<0:0:3>) updates(%14 : !VPURegMapped.Index<0:0:4>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %21 = VPUMI40XX.MappedInference dmas((%161, %20) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) invariants(%startIndexes#0 : !VPURegMapped.Index<0:0:0>) variants(%startIndexes#1 : !VPURegMapped.Index<0:0:0>) barriers(%10 : !VPURegMapped.Index<0:0:0>) dmaCount([[3, 1]]) invariantCount([2]) variantCount([2]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(5) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x14x14xf16, @DDR>
  }
}


// CHECK: [[VAL10:%.+]] = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}> <4, -1> -> !VPURegMapped.Index<0:0:0>
// CHECK: [[VAL11:%.+]] = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}>([[VAL10]] : !VPURegMapped.Index<0:0:0>) <0, -1> -> !VPURegMapped.Index<0:0:1>
// CHECK: [[VAL12:%.+]] = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 2 : ui8}>([[VAL11]] : !VPURegMapped.Index<0:0:1>) <1, -1> -> !VPURegMapped.Index<0:0:2>
// CHECK: [[VAL13:%.+]] = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}>([[VAL12]] : !VPURegMapped.Index<0:0:2>) <2, -1> -> !VPURegMapped.Index<0:0:3>
// CHECK: [[VAL14:%.+]] = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, isFinalBarrier, producer_count = 1 : ui8}>([[VAL13]] : !VPURegMapped.Index<0:0:3>) <3, -1> -> !VPURegMapped.Index<0:0:4>
// CHECK: [[startIndex:%.+]], [[startIndex:%.+]] = "VPURegMapped.ExecutionGroup"([[VAL11]], [[VAL13]])
// CHECK: [[VAL23:%.+]] = VPUMI40XX.DPUInvariant
// CHECK: [[VAL24:%.+]] = VPUMI40XX.DPUInvariant
// CHECK: [[VAL25:%.+]] = VPUMI40XX.DPUVariant
// CHECK: [[VAL26:%.+]] = VPUMI40XX.DPUVariant
// CHECK: "VPURegMapped.GroupYield"([[VAL23]], [[VAL25]], [[VAL24]], [[VAL26]]) <{operandSegmentSizes = array<i32: 2, 2>}> : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>) -> ()
// CHECK: VPURegMapped.FetchTask primary({{%.+}}#0 -> {{%.+}}#0) secondary({{%.+}}#1 -> {{%.+}}#1)

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
  %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, @DDR>
  %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1000x1x1xf16, @DDR>
  %2 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
  %3 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
  %4 = VPURT.DeclareBuffer <CMX_NN> [0] <1473536> -> memref<16xui32, [@CMX_NN, 0]>
  %5 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>
  %6 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>
  %7 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>
  %8 = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>
  %9 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>
  %10 = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>
  %11 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>
  %12 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>
  %13 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>
  %14 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>
  %15 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>
  %16 = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>
  %17 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>
  %18 = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>
  %19 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>
  %20 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>
  %21 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1000x1x1xf16, [@CMX_NN, 0]>
  %22 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>
  %23 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>
  // buffers for KernelParams of ActKernelInvocation ops in tile 0, list 1:
  %71 = VPURT.DeclareBuffer <CMX_NN> [0] <4048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>
  %72 = VPURT.DeclareBuffer <CMX_NN> [0] <6048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>
  %73 = VPURT.DeclareBuffer <CMX_NN> [0] <6048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>
  %74 = VPURT.DeclareBuffer <CMX_NN> [0] <4048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>
  %75 = VPURT.DeclareBuffer <CMX_NN> [0] <4048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>
  %76 = VPURT.DeclareBuffer <CMX_NN> [0] <6048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>
  %77 = VPURT.DeclareBuffer <CMX_NN> [0] <6048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>
  %78 = VPURT.DeclareBuffer <CMX_NN> [0] <4048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>
  // buffers for KernelParams of ActKernelInvocation ops in tile 1, list 1:
  %79 = VPURT.DeclareBuffer <CMX_NN> [1] <4048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>
  %80 = VPURT.DeclareBuffer <CMX_NN> [1] <6048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>
  %81 = VPURT.DeclareBuffer <CMX_NN> [1] <6048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>
  %82 = VPURT.DeclareBuffer <CMX_NN> [1] <4048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>
  %83 = VPURT.DeclareBuffer <CMX_NN> [1] <4048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>
  %84 = VPURT.DeclareBuffer <CMX_NN> [1] <6048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>
  %85 = VPURT.DeclareBuffer <CMX_NN> [1] <6048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>
  %86 = VPURT.DeclareBuffer <CMX_NN> [1] <4048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>

  %87 = VPURT.DeclareBuffer <CMX_NN> [0] <4048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>
  %88 = VPURT.DeclareBuffer <CMX_NN> [1] <4048> -> memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>

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
  %34 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%5 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>) outputs(%9 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:0:0>
  %97 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%71 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>) outputs(%72 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:1:0>
  %35 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%6 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>) outputs(%10 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:0:0>
  %98 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%79 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>) outputs(%80 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:1:0>
  %36 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%7 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>) outputs(%13 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:0:1>
  %99 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%73 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>) outputs(%74 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:1:1>
  %37 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%8 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>) outputs(%14 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:0:1>
  %100 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%81 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>) outputs(%82 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:1:1>
  %38 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%11 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>) outputs(%17 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:0:2>
  %101 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%75 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>) outputs(%76 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:1:2>
  %39 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%12 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>) outputs(%18 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:0:2>
  %102 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%83 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>) outputs(%84 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:1:2>
  %40 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%15 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>) outputs(%19 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:0:3>
  %103 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%77 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>) outputs(%78 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:1:3>
  %41 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%16 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>) outputs(%20 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:0:3>
  %104 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%85 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>) outputs(%86 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<1:1:3>
  %42 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, isStartBarrier, producer_count = 1 : ui8}> <0, -1> -> !VPURegMapped.Index<0:0:0>
  %43 = VPUMI40XX.ConfigureBarrier <{consumer_count = 4 : ui8, producer_count = 1 : ui8}>(%42 : !VPURegMapped.Index<0:0:0>) <1, -1> -> !VPURegMapped.Index<0:0:1>
  %44 = VPUMI40XX.ConfigureBarrier <{consumer_count = 4 : ui8, producer_count = 4 : ui8}>(%43 : !VPURegMapped.Index<0:0:1>) <2, -1> -> !VPURegMapped.Index<0:0:2>
  %45 = VPUMI40XX.ConfigureBarrier <{consumer_count = 4 : ui8, producer_count = 4 : ui8}>(%44 : !VPURegMapped.Index<0:0:2>) <3, -1> -> !VPURegMapped.Index<0:0:3>
  %46 = VPUMI40XX.ConfigureBarrier <{consumer_count = 4 : ui8, producer_count = 4 : ui8}>(%45 : !VPURegMapped.Index<0:0:3>) <4, -1> -> !VPURegMapped.Index<0:0:4>
  %47 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 4 : ui8}>(%46 : !VPURegMapped.Index<0:0:4>) <5, -1> -> !VPURegMapped.Index<0:0:5>
  %48 = VPUMI40XX.ConfigureBarrier <{consumer_count = 0 : ui8, isFinalBarrier, producer_count = 1 : ui8}>(%47 : !VPURegMapped.Index<0:0:5>) <6, -1> -> !VPURegMapped.Index<0:0:6>
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

  %491 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64, fetch_dma = #VPUIP.FetchDMAAttr<<SHAVE_ACT>, tile = 1 : i64, list = 1 : i64, fetchType = <DescriptorGroup>, group = 0 : i64>, wlmPage = -1 : i64}> inputs(%2 : memref<0x0x0x0xi32, @DDR>) outputs(%3 : memref<0x0x0x0xi32, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

  %492 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, fetch_dma = #VPUIP.FetchDMAAttr<<SHAVE_ACT>, tile = 1 : i64, list = 0 : i64, fetchType = <DescriptorGroup>, group = 0 : i64>, port = 0 : i64, wlmPage = -1 : i64}> inputs(%2 : memref<0x0x0x0xi32, @DDR>) outputs(%3 : memref<0x0x0x0xi32, @DDR>) previousDMA(%491 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>

  %493 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64, fetch_dma = #VPUIP.FetchDMAAttr<<SHAVE_ACT>, tile = 0 : i64, list = 1 : i64, fetchType = <DescriptorGroup>, group = 0 : i64>, wlmPage = -1 : i64}> inputs(%2 : memref<0x0x0x0xi32, @DDR>) outputs(%3 : memref<0x0x0x0xi32, @DDR>) previousDMA(%492 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>

   %494 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64, fetch_dma = #VPUIP.FetchDMAAttr<<SHAVE_ACT>, tile = 0 : i64, list = 0 : i64, fetchType = <DescriptorGroup>, group = 0 : i64>, wlmPage = -1 : i64}> inputs(%2 : memref<0x0x0x0xi32, @DDR>) outputs(%3 : memref<0x0x0x0xi32, @DDR>) previousDMA(%493 : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>

  %49 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64}> inputs(%2 : memref<0x0x0x0xi32, @DDR>) outputs(%3 : memref<0x0x0x0xi32, @DDR>) previousDMA(%494 : !VPURegMapped.Index<0:0:3>) updates(%42 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:4>

  %50 = VPUMI40XX.NNDMA <{is_out_of_order, port = 0 : i64}> inputs(%0 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, @DDR>) outputs(%22, %23, %87, %88 : memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>, memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>, memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 0]>, memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, [@CMX_NN, 1]>) previousDMA(%49 : !VPURegMapped.Index<0:0:4>) waits(%42 : !VPURegMapped.Index<0:0:0>) updates(%43 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x1000x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}, @DDR>, outputType = !VPUIP.DistributedBuffer<1x1000x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1000, 1, 1], [1, 1000, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1000, 1, 1], [1, 1000, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>>) -> !VPURegMapped.Index<0:0:5>

  %51 = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs(%21 : memref<1x1000x1x1xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x1000x1x1xf16, @DDR>) waits(%47 : !VPURegMapped.Index<0:0:5>) updates(%48 : !VPURegMapped.Index<0:0:6>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x1000x1x1xf16, [@CMX_NN, 0]>, outputType = memref<1x1000x1x1xf16, @DDR>>) -> !VPURegMapped.Index<0:1:0>
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
//CHECK-NEXT: [[START:%.+]]:[[NUM1:[0-9]+]], [[END:%.+]]:[[NUM2:[0-9]+]] = "VPURegMapped.ExecutionGroup"([[VAL43]], [[VAL47]])
//CHECK: [[START1:%.+]]:[[IGNORE:[0-9]+]], [[END1:%.+]]:[[IGNORE1:[0-9]+]] = "VPURegMapped.ExecutionGroup"([[VAL43]], [[VAL47]])
//CHECK: [[START2:%.+]]:[[IGNORE2:[0-9]+]], [[END2:%.+]]:[[IGNORE3:[0-9]+]] = "VPURegMapped.ExecutionGroup"([[VAL43]], [[VAL47]])
//CHECK: [[START3:%.+]]:[[IGNORE4:[0-9]+]], [[END3:%.+]]:[[IGNORE5:[0-9]+]] = "VPURegMapped.ExecutionGroup"([[VAL43]], [[VAL47]])
//CHECK:  [[VAL49:%.+]] = VPURegMapped.FetchTask primary([[START3]]#0 -> [[END3]]#0) secondary([[START3]]#1 -> [[END3]]#1) (<1:1:0> -> <1:1:3> : !VPURegMapped.Index<1:1:0> -> !VPURegMapped.Index<1:1:3>) -> <0:0:0>
//CHECK:  [[VAL50:%.+]] = VPURegMapped.FetchTask previousTask([[VAL49]] : !VPURegMapped.Index<0:0:0>) primary([[START2]]#0 -> [[END2]]#0) secondary([[START2]]#1 -> [[END2]]#1) (<1:0:0> -> <1:0:3> : !VPURegMapped.Index<1:0:0> -> !VPURegMapped.Index<1:0:3>) -> <0:0:1>
//CHECK:  [[VAL51:%.+]] = VPURegMapped.FetchTask previousTask([[VAL50]] : !VPURegMapped.Index<0:0:1>) primary([[START1]]#0 -> [[END1]]#0) secondary([[START1]]#1 -> [[END1]]#1) (<0:1:0> -> <0:1:3> : !VPURegMapped.Index<0:1:0> -> !VPURegMapped.Index<0:1:3>) -> <0:0:2>
//CHECK:  [[VAL52:%.+]] = VPURegMapped.FetchTask previousTask([[VAL51]] : !VPURegMapped.Index<0:0:2>) primary([[START]]#0 -> [[END]]#0) secondary([[START]]#1 -> [[END]]#1) (<0:0:0> -> <0:0:3> : !VPURegMapped.Index<0:0:0> -> !VPURegMapped.Index<0:0:3>) -> <0:0:3>

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 0.01269696927538105>
!qElemType2 = !quant.uniform<u8:f16, 0.0173492431640625:114>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @FetchWithEnqueueTarget {
  net.NetworkInfo entryPoint : @FetchWithEnqueueTarget inputsInfo : {
    DataInfo "Parameter_143" : tensor<1x3x224x224xf16>
  } outputsInfo : {
    DataInfo "Convolution_145" : tensor<1x64x56x56xf16>
  }
  func.func @FetchWithEnqueueTarget(%arg0: memref<1x3x224x224xf16, @DDR>, %arg1: memref<1x64x56x56xf16, @DDR>) -> memref<1x64x56x56xf16, @DDR> {
    %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <48832> -> memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>
    %3 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %4 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %5 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %6 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %7 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %8 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    %10 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x115x224xf16, [@CMX_NN, 0]>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x224x4x114x!qElemType, {order = #NWCH}, [@CMX_NN, 0]>
    %12 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x224x4x114x!qElemType, {order = #NWCH}, [@CMX_NN, 0]>
    %13 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %14 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %15 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %16 = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %17 = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %18 = VPURT.DeclareBuffer <CMX_NN> [0] <200704> -> memref<1x64x28x56xf16, [@CMX_NN, 0]>
    %19 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %20 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %21 = VPURT.DeclareBuffer <CMX_NN> [0] <257600> -> memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %22 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x16x114x224x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>
    %23 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, isStartBarrier, producer_count = 1 : ui8, wlmPage = 0 : i64}> <0, -1> -> !VPURegMapped.Index<0:0:0>
    %24 = VPUMI40XX.ConfigureBarrier <{consumer_count = 2 : ui8, producer_count = 2 : ui8, wlmPage = 0 : i64}>(%23 : !VPURegMapped.Index<0:0:0>) <1, -1> -> !VPURegMapped.Index<0:0:1>
    %25 = VPUMI40XX.ConfigureBarrier <{consumer_count = 2 : ui8, producer_count = 1 : ui8, wlmPage = 0 : i64}>(%24 : !VPURegMapped.Index<0:0:1>) <2, -1> -> !VPURegMapped.Index<0:0:2>
    %26 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 2 : ui8, wlmPage = 0 : i64}>(%25 : !VPURegMapped.Index<0:0:2>) <3, -1> -> !VPURegMapped.Index<0:0:3>
    %27 = VPUMI40XX.ConfigureBarrier <{consumer_count = 2 : ui8, producer_count = 1 : ui8, wlmPage = 1 : i64}>(%26, %24 : !VPURegMapped.Index<0:0:3>, !VPURegMapped.Index<0:0:1>) <0, -1> -> !VPURegMapped.Index<0:0:4>
    %28 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8, wlmPage = 1 : i64}>(%27 : !VPURegMapped.Index<0:0:4>) <1, -1> -> !VPURegMapped.Index<0:0:5>
    %29 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8, wlmPage = 1 : i64}>(%28 : !VPURegMapped.Index<0:0:5>) <2, -1> -> !VPURegMapped.Index<0:0:6>
    %30 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8, wlmPage = 1 : i64}>(%29 : !VPURegMapped.Index<0:0:6>) <3, -1> -> !VPURegMapped.Index<0:0:7>
    %31 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8, wlmPage = 2 : i64}>(%30, %27 : !VPURegMapped.Index<0:0:7>, !VPURegMapped.Index<0:0:4>) <0, -1> -> !VPURegMapped.Index<0:0:8>
    %32 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8, wlmPage = 2 : i64}>(%31 : !VPURegMapped.Index<0:0:8>) <1, -1> -> !VPURegMapped.Index<0:0:9>
    %33 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8, wlmPage = 2 : i64}>(%32 : !VPURegMapped.Index<0:0:9>) <2, -1> -> !VPURegMapped.Index<0:0:10>
    %34 = VPUMI40XX.ConfigureBarrier <{consumer_count = 2 : ui8, producer_count = 1 : ui8, wlmPage = 2 : i64}>(%33 : !VPURegMapped.Index<0:0:10>) <3, -1> -> !VPURegMapped.Index<0:0:11>
    %35 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 2 : ui8, wlmPage = 3 : i64}>(%34 : !VPURegMapped.Index<0:0:11>) <0, -1> -> !VPURegMapped.Index<0:0:12>
    %36 = VPUMI40XX.ConfigureBarrier <{consumer_count = 0 : ui8, isFinalBarrier, producer_count = 1 : ui8, wlmPage = 3 : i64}>(%35 : !VPURegMapped.Index<0:0:12>) <1, -1> -> !VPURegMapped.Index<0:0:13>
    %startIndexes:2, %endIndexes:2 = "VPURegMapped.ExecutionGroup"(%26, %24, %27) <{operandSegmentSizes = array<i32: 0, 2, 1>, resultSegmentSizes = array<i32: 2, 2>, task_type = #VPURegMapped.task_type<DPUInvariant>}> ({
      %50 = VPUMI40XX.DPUInvariant <{clean_after = 0 : ui64, is_permute_quantize, is_superdense, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, wlmPage = 0 : i64}> input(%20 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%19 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%11 : memref<1x224x4x114x!qElemType, {order = #NWCH}, [@CMX_NN, 0]>) waits(%26, %24 : !VPURegMapped.Index<0:0:3>, !VPURegMapped.Index<0:0:1>) updates(%27 : !VPURegMapped.Index<0:0:4>) enqueueBarrier(%23 : !VPURegMapped.Index<0:0:0>) -> <0:0:0> PPE : {
        VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
      }
      %51 = VPUMI40XX.DPUVariant calls(%50 : <0:0:0>) weights(%19 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [113, 2, 223], inEnd = [113, 2, 223], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 0 : i64}> -> <0:0:0>
      %52 = VPUMI40XX.DPUVariant previousTask(%51 : !VPURegMapped.Index<0:0:0>) calls(%50 : <0:0:0>) weights(%19 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [113, 2, 223], inEnd = [113, 2, 223], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 0 : i64}> -> <0:0:1>
      %53 = VPUMI40XX.DPUVariant previousTask(%52 : !VPURegMapped.Index<0:0:1>) calls(%50 : <0:0:0>) weights(%19 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [113, 2, 223], inEnd = [113, 2, 223], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 0 : i64}> -> <0:0:2>
      %54 = VPUMI40XX.DPUVariant previousTask(%53 : !VPURegMapped.Index<0:0:2>) calls(%50 : <0:0:0>) weights(%19 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) {lastSecondaryTaskInExecutionGroup} <{end = [113, 2, 223], inEnd = [113, 2, 223], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 0 : i64}> -> <0:0:3>
      "VPURegMapped.GroupYield"(%50, %51, %50, %54) <{operandSegmentSizes = array<i32: 2, 2>}> : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:3>) -> ()
    }) : (!VPURegMapped.Index<0:0:3>, !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:4>) -> (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:3>)
    %startIndexes_0:2, %endIndexes_1:2 = "VPURegMapped.ExecutionGroup"(%endIndexes#0, %endIndexes#1, %28, %29) <{operandSegmentSizes = array<i32: 2, 1, 1>, resultSegmentSizes = array<i32: 2, 2>, task_type = #VPURegMapped.task_type<DPUInvariant>}> ({
    ^bb0(%arg2: !VPURegMapped.Index<0:0:0>, %arg3: !VPURegMapped.Index<0:0:3>):
      %50 = VPUMI40XX.DPUInvariant <{clean_after = 0 : ui64, is_permute_quantize, is_superdense, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, wlmPage = 1 : i64}> previousTask(%arg2 : !VPURegMapped.Index<0:0:0>) input(%20 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%19 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%12 : memref<1x224x4x114x!qElemType, {order = #NWCH}, [@CMX_NN, 0]>) waits(%28 : !VPURegMapped.Index<0:0:5>) updates(%29 : !VPURegMapped.Index<0:0:6>) enqueueBarrier(%23 : !VPURegMapped.Index<0:0:0>) -> <0:0:1> PPE : {
        VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
      }
      %51 = VPUMI40XX.DPUVariant previousTask(%arg3 : !VPURegMapped.Index<0:0:3>) calls(%50 : <0:0:1>) weights(%19 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [113, 2, 223], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 1 : i64}> -> <0:0:4>
      %52 = VPUMI40XX.DPUVariant previousTask(%51 : !VPURegMapped.Index<0:0:4>) calls(%50 : <0:0:1>) weights(%19 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [113, 2, 223], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 1 : i64}> -> <0:0:5>
      %53 = VPUMI40XX.DPUVariant previousTask(%52 : !VPURegMapped.Index<0:0:5>) calls(%50 : <0:0:1>) weights(%19 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [113, 2, 223], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 1 : i64}> -> <0:0:6>
      %54 = VPUMI40XX.DPUVariant previousTask(%53 : !VPURegMapped.Index<0:0:6>) calls(%50 : <0:0:1>) weights(%19 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) {lastSecondaryTaskInExecutionGroup} <{end = [113, 2, 223], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 1 : i64}> -> <0:0:7>
      "VPURegMapped.GroupYield"(%50, %51, %50, %54) <{operandSegmentSizes = array<i32: 2, 2>}> : (!VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:4>, !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:7>) -> ()
    }) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:3>, !VPURegMapped.Index<0:0:5>, !VPURegMapped.Index<0:0:6>) -> (!VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:4>, !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:7>)
    %startIndexes_2:2, %endIndexes_3:2 = "VPURegMapped.ExecutionGroup"(%endIndexes_1#0, %endIndexes_1#1, %30, %27, %31) <{operandSegmentSizes = array<i32: 2, 2, 1>, resultSegmentSizes = array<i32: 2, 2>, task_type = #VPURegMapped.task_type<DPUInvariant>}> ({
    ^bb0(%arg2: !VPURegMapped.Index<0:0:1>, %arg3: !VPURegMapped.Index<0:0:7>):
      %50 = VPUMI40XX.DPUInvariant <{clean_after = 0 : ui64, cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, wlmPage = 1 : i64}> previousTask(%arg2 : !VPURegMapped.Index<0:0:1>) input(%22 : memref<1x16x114x224x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>) weights(%21 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%14 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) waits(%30, %27 : !VPURegMapped.Index<0:0:7>, !VPURegMapped.Index<0:0:4>) updates(%31 : !VPURegMapped.Index<0:0:8>) enqueueBarrier(%23 : !VPURegMapped.Index<0:0:0>) -> <0:0:2> PPE : {
        VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
      }
      %51 = VPUMI40XX.DPUVariant previousTask(%arg3 : !VPURegMapped.Index<0:0:7>) calls(%50 : <0:0:2>) weights(%21 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [111, 55, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 1 : i64}> -> <0:0:8>
      %52 = VPUMI40XX.DPUVariant previousTask(%51 : !VPURegMapped.Index<0:0:8>) calls(%50 : <0:0:2>) weights(%21 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [111, 55, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 1 : i64}> -> <0:0:9>
      %53 = VPUMI40XX.DPUVariant previousTask(%52 : !VPURegMapped.Index<0:0:9>) calls(%50 : <0:0:2>) weights(%21 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [111, 55, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 1 : i64}> -> <0:0:10>
      %54 = VPUMI40XX.DPUVariant previousTask(%53 : !VPURegMapped.Index<0:0:10>) calls(%50 : <0:0:2>) weights(%21 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) {lastSecondaryTaskInExecutionGroup} <{end = [111, 55, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 1 : i64}> -> <0:0:11>
      "VPURegMapped.GroupYield"(%50, %51, %50, %54) <{operandSegmentSizes = array<i32: 2, 2>}> : (!VPURegMapped.Index<0:0:2>, !VPURegMapped.Index<0:0:8>, !VPURegMapped.Index<0:0:2>, !VPURegMapped.Index<0:0:11>) -> ()
    }) : (!VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:7>, !VPURegMapped.Index<0:0:7>, !VPURegMapped.Index<0:0:4>, !VPURegMapped.Index<0:0:8>) -> (!VPURegMapped.Index<0:0:2>, !VPURegMapped.Index<0:0:8>, !VPURegMapped.Index<0:0:2>, !VPURegMapped.Index<0:0:11>)
    %startIndexes_4:2, %endIndexes_5:2 = "VPURegMapped.ExecutionGroup"(%endIndexes_3#0, %endIndexes_3#1, %32, %33) <{operandSegmentSizes = array<i32: 2, 1, 1>, resultSegmentSizes = array<i32: 2, 2>, task_type = #VPURegMapped.task_type<DPUInvariant>}> ({
    ^bb0(%arg2: !VPURegMapped.Index<0:0:2>, %arg3: !VPURegMapped.Index<0:0:11>):
      %50 = VPUMI40XX.DPUInvariant <{clean_after = 0 : ui64, cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, wlmPage = 2 : i64}> previousTask(%arg2 : !VPURegMapped.Index<0:0:2>) input(%22 : memref<1x16x114x224x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>) weights(%21 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%15 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) waits(%32 : !VPURegMapped.Index<0:0:9>) updates(%33 : !VPURegMapped.Index<0:0:10>) enqueueBarrier(%24 : !VPURegMapped.Index<0:0:1>) -> <0:0:3> PPE : {
        VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
      }
      %51 = VPUMI40XX.DPUVariant previousTask(%arg3 : !VPURegMapped.Index<0:0:11>) calls(%50 : <0:0:3>) weights(%21 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [111, 55, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:12>
      %52 = VPUMI40XX.DPUVariant previousTask(%51 : !VPURegMapped.Index<0:0:12>) calls(%50 : <0:0:3>) weights(%21 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [111, 55, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:13>
      %53 = VPUMI40XX.DPUVariant previousTask(%52 : !VPURegMapped.Index<0:0:13>) calls(%50 : <0:0:3>) weights(%21 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [111, 55, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:14>
      %54 = VPUMI40XX.DPUVariant previousTask(%53 : !VPURegMapped.Index<0:0:14>) calls(%50 : <0:0:3>) weights(%21 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) {lastSecondaryTaskInExecutionGroup} <{end = [111, 55, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:15>
      "VPURegMapped.GroupYield"(%50, %51, %50, %54) <{operandSegmentSizes = array<i32: 2, 2>}> : (!VPURegMapped.Index<0:0:3>, !VPURegMapped.Index<0:0:12>, !VPURegMapped.Index<0:0:3>, !VPURegMapped.Index<0:0:15>) -> ()
    }) : (!VPURegMapped.Index<0:0:2>, !VPURegMapped.Index<0:0:11>, !VPURegMapped.Index<0:0:9>, !VPURegMapped.Index<0:0:10>) -> (!VPURegMapped.Index<0:0:3>, !VPURegMapped.Index<0:0:12>, !VPURegMapped.Index<0:0:3>, !VPURegMapped.Index<0:0:15>)
    %startIndexes_6:2, %endIndexes_7:2 = "VPURegMapped.ExecutionGroup"(%endIndexes_5#0, %endIndexes_5#1, %34, %35) <{operandSegmentSizes = array<i32: 2, 1, 1>, resultSegmentSizes = array<i32: 2, 2>, task_type = #VPURegMapped.task_type<DPUInvariant>}> ({
    ^bb0(%arg2: !VPURegMapped.Index<0:0:3>, %arg3: !VPURegMapped.Index<0:0:15>):
      %50 = VPUMI40XX.DPUInvariant <{clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, start_after = 0 : ui64, wlmPage = 2 : i64}> previousTask(%arg2 : !VPURegMapped.Index<0:0:3>) input(%13 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) outputs(%16 : memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) waits(%34 : !VPURegMapped.Index<0:0:11>) updates(%35 : !VPURegMapped.Index<0:0:12>) enqueueBarrier(%27 : !VPURegMapped.Index<0:0:4>) -> <0:0:4> PPE : {
        VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
      }
      %51 = VPUMI40XX.DPUVariant previousTask(%arg3 : !VPURegMapped.Index<0:0:15>) calls(%50 : <0:0:4>) <{end = [55, 27, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:16>
      %52 = VPUMI40XX.DPUVariant previousTask(%51 : !VPURegMapped.Index<0:0:16>) calls(%50 : <0:0:4>) <{end = [55, 27, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:17>
      %53 = VPUMI40XX.DPUVariant previousTask(%52 : !VPURegMapped.Index<0:0:17>) calls(%50 : <0:0:4>) <{end = [55, 27, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:18>
      %54 = VPUMI40XX.DPUVariant previousTask(%53 : !VPURegMapped.Index<0:0:18>) calls(%50 : <0:0:4>) {lastSecondaryTaskInExecutionGroup} <{end = [55, 27, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:19>
      "VPURegMapped.GroupYield"(%50, %51, %50, %54) <{operandSegmentSizes = array<i32: 2, 2>}> : (!VPURegMapped.Index<0:0:4>, !VPURegMapped.Index<0:0:16>, !VPURegMapped.Index<0:0:4>, !VPURegMapped.Index<0:0:19>) -> ()
    }) : (!VPURegMapped.Index<0:0:3>, !VPURegMapped.Index<0:0:15>, !VPURegMapped.Index<0:0:11>, !VPURegMapped.Index<0:0:12>) -> (!VPURegMapped.Index<0:0:4>, !VPURegMapped.Index<0:0:16>, !VPURegMapped.Index<0:0:4>, !VPURegMapped.Index<0:0:19>)
    %startIndexes_8:2, %endIndexes_9:2 = "VPURegMapped.ExecutionGroup"(%endIndexes_7#0, %endIndexes_7#1, %34, %35) <{operandSegmentSizes = array<i32: 2, 1, 1>, resultSegmentSizes = array<i32: 2, 2>, task_type = #VPURegMapped.task_type<DPUInvariant>}> ({
    ^bb0(%arg2: !VPURegMapped.Index<0:0:4>, %arg3: !VPURegMapped.Index<0:0:19>):
      %50 = VPUMI40XX.DPUInvariant <{clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, start_after = 0 : ui64, wlmPage = 2 : i64}> previousTask(%arg2 : !VPURegMapped.Index<0:0:4>) input(%13 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) outputs(%17 : memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) waits(%34 : !VPURegMapped.Index<0:0:11>) updates(%35 : !VPURegMapped.Index<0:0:12>) enqueueBarrier(%27 : !VPURegMapped.Index<0:0:4>) -> <0:0:5> PPE : {
        VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
      }
      %51 = VPUMI40XX.DPUVariant previousTask(%arg3 : !VPURegMapped.Index<0:0:19>) calls(%50 : <0:0:5>) <{end = [55, 27, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:20>
      %52 = VPUMI40XX.DPUVariant previousTask(%51 : !VPURegMapped.Index<0:0:20>) calls(%50 : <0:0:5>) <{end = [55, 27, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:21>
      %53 = VPUMI40XX.DPUVariant previousTask(%52 : !VPURegMapped.Index<0:0:21>) calls(%50 : <0:0:5>) <{end = [55, 27, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:22>
      %54 = VPUMI40XX.DPUVariant previousTask(%53 : !VPURegMapped.Index<0:0:22>) calls(%50 : <0:0:5>) {lastSecondaryTaskInExecutionGroup} <{end = [55, 27, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:23>
      "VPURegMapped.GroupYield"(%50, %51, %50, %54) <{operandSegmentSizes = array<i32: 2, 2>}> : (!VPURegMapped.Index<0:0:5>, !VPURegMapped.Index<0:0:20>, !VPURegMapped.Index<0:0:5>, !VPURegMapped.Index<0:0:23>) -> ()
    }) : (!VPURegMapped.Index<0:0:4>, !VPURegMapped.Index<0:0:19>, !VPURegMapped.Index<0:0:11>, !VPURegMapped.Index<0:0:12>) -> (!VPURegMapped.Index<0:0:5>, !VPURegMapped.Index<0:0:20>, !VPURegMapped.Index<0:0:5>, !VPURegMapped.Index<0:0:23>)
    %37 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64, wlmPage = 0 : i64}> inputs(%3 : memref<0x0x0x0xi32, @DDR>) outputs(%4 : memref<0x0x0x0xi32, @DDR>) updates(%23 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %38 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64, wlmPage = 0 : i64}> inputs(%3 : memref<0x0x0x0xi32, @DDR>) outputs(%4 : memref<0x0x0x0xi32, @DDR>) previousDMA(%37 : !VPURegMapped.Index<0:0:0>) waits(%23 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %39 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, fetch_dma = #VPUIP.FetchDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, fetchType = <DescriptorGroup>, group = 0 : i64>, port = 0 : i64, wlmPage = 0 : i64}> inputs(%7 : memref<0x0x0x0xi32, @DDR>) outputs(%8 : memref<0x0x0x0xi32, @DDR>) previousDMA(%38 : !VPURegMapped.Index<0:0:1>) updates(%24 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %40 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, fetch_dma = #VPUIP.FetchDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, fetchType = <DescriptorGroup>, group = 1 : i64>, port = 0 : i64, wlmPage = 0 : i64}> inputs(%7 : memref<0x0x0x0xi32, @DDR>) outputs(%8 : memref<0x0x0x0xi32, @DDR>) previousDMA(%39 : !VPURegMapped.Index<0:0:2>) updates(%24 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>
    %41 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64, wlmPage = 0 : i64}> inputs(%5 : memref<0x0x0x0xi32, @DDR>) outputs(%6 : memref<0x0x0x0xi32, @DDR>) previousDMA(%40 : !VPURegMapped.Index<0:0:3>) waits(%24 : !VPURegMapped.Index<0:0:1>) updates(%25 : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:4>
    %42 = VPUMI40XX.NNDMA <{port = 0 : i64, wlmPage = 0 : i64}> inputs(%0 : memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%9 : memref<1x3x114x224xf16, [@CMX_NN, 0]>) previousDMA(%41 : !VPURegMapped.Index<0:0:4>) waits(%25 : !VPURegMapped.Index<0:0:2>) updates(%26 : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>, outputType = memref<1x3x114x224xf16, [@CMX_NN, 0]>>) -> !VPURegMapped.Index<0:0:5>
    %43 = VPUMI40XX.NNDMA <{port = 0 : i64, wlmPage = 0 : i64}> inputs(%1 : memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%10 : memref<1x3x115x224xf16, [@CMX_NN, 0]>) previousDMA(%42 : !VPURegMapped.Index<0:0:5>) waits(%25 : !VPURegMapped.Index<0:0:2>) updates(%26 : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>, outputType = memref<1x3x115x224xf16, [@CMX_NN, 0]>>) -> !VPURegMapped.Index<0:0:6>
    %44 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, fetch_dma = #VPUIP.FetchDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, fetchType = <DescriptorGroup>, group = 2 : i64>, port = 0 : i64, wlmPage = 1 : i64}> inputs(%7 : memref<0x0x0x0xi32, @DDR>) outputs(%8 : memref<0x0x0x0xi32, @DDR>) previousDMA(%43 : !VPURegMapped.Index<0:0:6>) waits(%27 : !VPURegMapped.Index<0:0:4>) updates(%28 : !VPURegMapped.Index<0:0:5>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:7>
    %45 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, fetch_dma = #VPUIP.FetchDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, fetchType = <DescriptorGroup>, group = 3 : i64>, port = 0 : i64, wlmPage = 1 : i64}> inputs(%7 : memref<0x0x0x0xi32, @DDR>) outputs(%8 : memref<0x0x0x0xi32, @DDR>) previousDMA(%44 : !VPURegMapped.Index<0:0:7>) waits(%29 : !VPURegMapped.Index<0:0:6>) updates(%30 : !VPURegMapped.Index<0:0:7>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:8>
    %46 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, fetch_dma = #VPUIP.FetchDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, fetchType = <DescriptorGroup>, group = 4 : i64>, port = 0 : i64, wlmPage = 2 : i64}> inputs(%7 : memref<0x0x0x0xi32, @DDR>) outputs(%8 : memref<0x0x0x0xi32, @DDR>) previousDMA(%45 : !VPURegMapped.Index<0:0:8>) waits(%31 : !VPURegMapped.Index<0:0:8>) updates(%32 : !VPURegMapped.Index<0:0:9>) enqueueBarrier(%24 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:9>
    %47 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, fetch_dma = #VPUIP.FetchDMAAttr<<DPU>, tile = 0 : i64, list = 0 : i64, fetchType = <DescriptorGroup>, group = 5 : i64>, port = 0 : i64, wlmPage = 2 : i64}> inputs(%7 : memref<0x0x0x0xi32, @DDR>) outputs(%8 : memref<0x0x0x0xi32, @DDR>) previousDMA(%46 : !VPURegMapped.Index<0:0:9>) waits(%33 : !VPURegMapped.Index<0:0:10>) updates(%34 : !VPURegMapped.Index<0:0:11>) enqueueBarrier(%24 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:10>
    %48 = VPUMI40XX.NNDMA <{port = 0 : i64, wlmPage = 3 : i64}> inputs(%18 : memref<1x64x28x56xf16, [@CMX_NN, 0]>) outputs(%2 : memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>) waits(%35 : !VPURegMapped.Index<0:0:12>) updates(%36 : !VPURegMapped.Index<0:0:13>) enqueueBarrier(%27 : !VPURegMapped.Index<0:0:4>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x64x28x56xf16, [@CMX_NN, 0]>, outputType = memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>>) -> !VPURegMapped.Index<0:1:0>
    %49 = VPUMI40XX.MappedInference dmas((%37, %48) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) invariants(%startIndexes#0 : !VPURegMapped.Index<0:0:0>) variants(%startIndexes#1 : !VPURegMapped.Index<0:0:0>) barriers(%23 : !VPURegMapped.Index<0:0:0>) dmaCount([[11, 1], [0, 0]]) invariantCount([6, 0, 0, 0, 0, 0]) variantCount([24, 0, 0, 0, 0, 0]) actKernelRangesCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) actKernelInvocationsCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) mediaCount(0) barrierCount(14) -> !VPURegMapped.Index<0:0:0>
    VPUMI40XX.OpRanges types([#VPURegMapped.task_type<DMA>, #VPURegMapped.task_type<DPUInvariant>, #VPURegMapped.task_type<DPUVariant>, #VPURegMapped.task_type<DMA>]) begins(%37, %startIndexes#0, %startIndexes#1, %48 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>) ends(%47, %startIndexes_8#0, %endIndexes_9#1, %48 : !VPURegMapped.Index<0:0:10>, !VPURegMapped.Index<0:0:5>, !VPURegMapped.Index<0:0:23>, !VPURegMapped.Index<0:1:0>)
  }

  // CHECK: [[BAR_START:%.+]] = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, isStartBarrier, producer_count = 1 : ui8, wlmPage = 0 : i64}> <0, -1> -> !VPURegMapped.Index<0:0:0>
  // CHECK: [[ENQ_BAR:%.+]] = VPUMI40XX.ConfigureBarrier <{consumer_count = 2 : ui8, producer_count = 2 : ui8, wlmPage = 0 : i64}>([[BAR_START]] : !VPURegMapped.Index<0:0:0>) <1, -1> -> !VPURegMapped.Index<0:0:1>
  // Fetch Group 0
  // CHECK: VPURegMapped.FetchTask
  // Fetch Group 1
  // CHECK: VPURegMapped.FetchTask
  // Fetch Group 2
  // CHECK: VPURegMapped.FetchTask
  // Fetch Group 3
  // CHECK: VPURegMapped.FetchTask
  // Fetch Group 4
  // CHECK: VPURegMapped.FetchTask
  // CHECK-SAME: enqueueBarrier([[ENQ_BAR]] : !VPURegMapped.Index<0:0:1>)
  // Fetch Group 5
  // CHECK: VPURegMapped.FetchTask
  // CHECK-SAME: enqueueBarrier([[ENQ_BAR]] : !VPURegMapped.Index<0:0:1>)
}
