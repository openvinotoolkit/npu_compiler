//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --unroll-fetch-ops %s | FileCheck %s
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
    %15 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
    %16 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
    %17 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
    %18 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
    %19 = VPUMI40XX.DPUInvariant <{clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64}>
    taskLocation(%15 : !VPURegMapped.Index<0:0:0>) input(%6 : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, {order = #NWCH}, [@CMX_NN, 0]>)
    waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%12 : !VPURegMapped.Index<0:0:2>) -> <0:0:0> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %20 = VPUMI40XX.DPUInvariant <{clean_after = 3 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3],
    kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 4 : ui64}>
    taskLocation(%16 : !VPURegMapped.Index<0:0:1>) previousTask(%19 : !VPURegMapped.Index<0:0:0>) input(%7 : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    weights(%9 : memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>)
    waits(%12 : !VPURegMapped.Index<0:0:2>) updates(%13 : !VPURegMapped.Index<0:0:3>) -> <0:0:1> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %21 = VPUMI40XX.DPUVariant taskLocation(%17 : !VPURegMapped.Index<0:0:0>) calls(%19 : <0:0:0>) weights(%6 : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}> -> <0:0:0>
    %22 = VPUMI40XX.DPUVariant taskLocation(%18 : !VPURegMapped.Index<0:0:1>) previousTask(%21 : !VPURegMapped.Index<0:0:0>) calls(%20 : <0:0:1>) weights(%9 : memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) <{end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}> -> <0:0:1>
    %23 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %24 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %25 = VPURegMapped.FetchTask primary(%19 -> %20) secondary(%21 -> %22) (<0:0:0> -> <0:0:1> : !VPURegMapped.Index<0:0:0> -> !VPURegMapped.Index<0:0:1>) -> <0:0:0>
    %26 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 0 : i32, srcWidth = 0 : i32, srcStride = 0 : i32, srcPlaneStride = 0 : i32, dstWidth = 0 : i32, dstStride = 0 : i32, dstPlaneStride = 0 : i32>, port = 1 : i64}> inputs(%23 : memref<1x1x1x1xi32, @DDR>) outputs(%24 : memref<1x1x1x1xi32, @DDR>) previousDMA(%25 : !VPURegMapped.Index<0:0:0>) updates(%10 : !VPURegMapped.Index<0:0:0>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %27 = VPUMI40XX.NNDMA <{port = 1 : i64}> inputs(%0 : memref<1x16x16x16xf16, @DDR>) outputs(%2 : memref<1x16x16x16xf16, [@CMX_NN, 0]>) previousDMA(%26 : !VPURegMapped.Index<0:0:1>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%11 : !VPURegMapped.Index<0:0:1>) start_after(2) clean_after(1) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %28 = VPUMI40XX.NNDMA <{is_out_of_order, port = 1 : i64}> inputs(%cst : memref<1x1x1x4864xui8>) outputs(%5 : memref<1x1x1x4864xui8, [@CMX_NN, 0]>) previousDMA(%27 : !VPURegMapped.Index<0:0:2>) updates(%12 : !VPURegMapped.Index<0:0:2>) start_after(3) clean_after(2) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>
    %29 = VPUMI40XX.NNDMA <{port = 1 : i64}> inputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x16x14x14xf16, @DDR>) waits(%13 : !VPURegMapped.Index<0:0:3>) updates(%14 : !VPURegMapped.Index<0:0:4>) start_after(5) clean_after(4) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %30 = VPURegMapped.Enqueue at(%10 : !VPURegMapped.Index<0:0:0>) (%21 -> %21 : <0:0:0> -> <0:0:0>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<DPUVariant>}
    %31 = VPURegMapped.Enqueue previousTaskIdx(%30 : !VPURegMapped.Index<0:0:0>) at(%10 : !VPURegMapped.Index<0:0:0>) (%22 -> %22 : <0:0:1> -> <0:0:1>) -> !VPURegMapped.Index<0:0:1> {taskType = #VPURegMapped.task_type<DPUVariant>}
    %32 = VPUMI40XX.MappedInference dmas((%25, %29) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) invariants(%19 : !VPURegMapped.Index<0:0:0>) variants(%21 : !VPURegMapped.Index<0:0:0>) barriers(%10 : !VPURegMapped.Index<0:0:0>) workItemTasks(%30 : !VPURegMapped.Index<0:0:0>) dmaCount([[4, 1]]) invariantCount([2]) variantCount([2]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(5) workItemCount(2) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x14x14xf16, @DDR>
  }
}

// CHECK-NOT: VPURegMapped.FetchTask
//CHECK: [[VAL15:%.+]] = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
//CHECK: [[VAL16:%.+]] = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
//CHECK: [[VAL17:%.+]] = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
//CHECK: [[VAL18:%.+]] = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
//CHECK: [[VAL19:%.+]] = VPUMI40XX.DPUInvariant
//CHECK: [[VAL20:%.+]] = VPUMI40XX.DPUInvariant
//CHECK: [[VAL21:%.+]] = VPUMI40XX.DPUVariant taskLocation([[VAL17]] : !VPURegMapped.Index<0:0:0>) calls([[VAL19]] : <0:0:0>)
//CHECK: [[VAL22:%.+]] = VPUMI40XX.DPUVariant taskLocation([[VAL18]] : !VPURegMapped.Index<0:0:1>) previousTask([[VAL21]] : !VPURegMapped.Index<0:0:0>) calls([[VAL20]] : <0:0:1>)
//CHECK: VPURegMapped.ViewTaskRange([[VAL19]] -> [[VAL20]] : <0:0:0> -> <0:0:1>) -> memref<2x{{.+}}xui8>
//CHECK: VPURegMapped.ViewTaskRange([[VAL15]] -> [[VAL16]] : <0:0:0> -> <0:0:1>) -> memref<2x{{.+}}xui8, [@CMX_NN, 0]>
//CHECK: VPURegMapped.ViewTaskRange([[VAL21]] -> [[VAL22]] : <0:0:0> -> <0:0:1>) -> memref<2x224xui8>
//CHECK: VPURegMapped.ViewTaskRange([[VAL17]] -> [[VAL18]] : <0:0:0> -> <0:0:1>) -> memref<2x224xui8, [@CMX_NN, 0]>


// -----

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
    DataInfo "input" : tensor<1x1000x1x1xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1000x1x1xf16>
  }
  func.func @main(%arg0: memref<1x1000x1x1xf16, @DDR>, %arg1: memref<1x1000x1x1xf16, @DDR>) -> memref<1x1000x1x1xf16, @DDR> {
  %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<3x3844xf16, @DDR>
  %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x62x62xf16, @DDR>
  %2 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
  %3 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
  %4 = VPURT.DeclareBuffer <CMX_NN> [0] <1473536> -> memref<16xui32, [@CMX_NN, 0]>
  %5 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<3844x3xf16, [@CMX_NN, 0]>
  %6 = VPURT.DeclareBuffer <CMX_NN> [0] <23104> -> memref<3844x3xf16, [@CMX_NN, 0]>
  %7 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x62x62xf16, [@CMX_NN, 0]>
  %8 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<3x3844xf16, [@CMX_NN, 0]>
  %9 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x32x62xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [11532, 1, 186, 3]}, [@CMX_NN, 0]>
  %10 = VPURT.DeclareBuffer <CMX_NN> [0] <23104> -> memref<1x3x32x62xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [11532, 1, 186, 3]}, [@CMX_NN, 0]>
  %11 = VPURT.DeclareBuffer <CMX_NN> [0] <11904> -> memref<1x3x30x62xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [11532, 1, 186, 3]}, [@CMX_NN, 0]>
  %12 = VPURT.DeclareBuffer <CMX_NN> [0] <35008> -> memref<1x3x30x62xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [11532, 1, 186, 3]}, [@CMX_NN, 0]>
  %13 = VPUMI40XX.DeclareKernelText kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
  %14 = VPUMI40XX.DeclareKernelEntry kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
  %15 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
  %16 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:1>
  %17 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%9 : memref<1x3x32x62xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [11532, 1, 186, 3]}, [@CMX_NN, 0]>) outputs(%10 : memref<1x3x32x62xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [11532, 1, 186, 3]}, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:0:0>
  %18 = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs(%11 : memref<1x3x30x62xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [11532, 1, 186, 3]}, [@CMX_NN, 0]>) outputs(%12 : memref<1x3x30x62xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [11532, 1, 186, 3]}, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:0:1>
  %19 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, isStartBarrier, producer_count = 1 : ui8}> <0, -1> -> !VPURegMapped.Index<0:0:0>
  %20 = VPUMI40XX.ConfigureBarrier <{consumer_count = 2 : ui8, producer_count = 1 : ui8}>(%19 : !VPURegMapped.Index<0:0:0>) <1, -1> -> !VPURegMapped.Index<0:0:1>
  %21 = VPUMI40XX.ConfigureBarrier <{consumer_count = 2 : ui8, producer_count = 2 : ui8}>(%20 : !VPURegMapped.Index<0:0:1>) <2, -1> -> !VPURegMapped.Index<0:0:2>
  %22 = VPUMI40XX.ConfigureBarrier <{consumer_count = 0 : ui8, isFinalBarrier, producer_count = 1 : ui8}>(%21 : !VPURegMapped.Index<0:0:2>) <3, -1> -> !VPURegMapped.Index<0:0:3>
  %53 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:30>
  %54 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:31>
  %117 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:0:30>
  %118 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:0:31>
  %151 = VPUMI40XX.ActKernelRange taskLocation(%117 : !VPURegMapped.Index<0:0:30>) kernel_text_index(%13 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%15 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%14 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
  %152 = VPUMI40XX.ActKernelRange taskLocation(%118 : !VPURegMapped.Index<0:0:31>) previousTask(%151 : !VPURegMapped.Index<0:0:0>) kernel_text_index(%13 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%16 : !VPURegMapped.Index<0:0:1>) kernel_entry_index(%14 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:1>
  %153 = VPUMI40XX.ActKernelInvocation taskLocation(%53 : !VPURegMapped.Index<0:0:30>) range_index(%151 : <0:0:0>) kernel_params(%17 : <0:0:0>) waits(%20 : !VPURegMapped.Index<0:0:1>) updates(%21 : !VPURegMapped.Index<0:0:2>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
  %154 = VPUMI40XX.ActKernelInvocation {lastSecondaryTaskInExecutionGroup} taskLocation(%54 : !VPURegMapped.Index<0:0:31>) previousTask(%153 : !VPURegMapped.Index<0:0:0>) range_index(%152 : <0:0:1>) kernel_params(%18 : <0:0:1>) waits(%20 : !VPURegMapped.Index<0:0:1>) updates(%21 : !VPURegMapped.Index<0:0:2>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
  %155 = VPURegMapped.FetchTask primary(%151 -> %152) secondary(%153 -> %154) (<0:0:0> -> <0:0:1> : !VPURegMapped.Index<0:0:0> -> !VPURegMapped.Index<0:0:1>) -> <0:0:0> {associated_execution_group_index = 0 : ui64, associated_task_type = #VPURegMapped.task_type<ActKernelRange>, associated_tile_index = 0 : ui64}
  %156 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64}> inputs(%2 : memref<0x0x0x0xi32, @DDR>) outputs(%3 : memref<0x0x0x0xi32, @DDR>) previousDMA(%155 : !VPURegMapped.Index<0:0:0>) updates(%19 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
  %157 = VPUMI40XX.NNDMA <{allow_different_in_out_shapes, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 3 : i64, len = 7688 : i64, srcWidth = 7688 : i64, srcStride = 2 : i64, srcPlaneStride = 7688 : i64, dstWidth = 2 : i64, dstStride = 6 : i64, dstPlaneStride = 2 : i64>, is_out_of_order, port = 0 : i64}> inputs(%0 : memref<3x3844xf16, @DDR>) outputs(%5 : memref<3844x3xf16, [@CMX_NN, 0]>) previousDMA(%156 : !VPURegMapped.Index<0:0:1>) waits(%19 : !VPURegMapped.Index<0:0:0>) updates(%20 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
  %158 = VPUMI40XX.NNDMA <{allow_different_in_out_shapes, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 3844 : i64, len = 6 : i64, srcWidth = 6 : i64, srcStride = 2 : i64, srcPlaneStride = 6 : i64, dstWidth = 2 : i64, dstStride = 7688 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64}> inputs(%6 : memref<3844x3xf16, [@CMX_NN, 0]>) outputs(%8 : memref<3x3844xf16, [@CMX_NN, 0]>) waits(%21 : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
  %159 = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs(%7 : memref<1x3x62x62xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x3x62x62xf16, @DDR>) previousDMA(%158 : !VPURegMapped.Index<0:1:0>) waits(%21 : !VPURegMapped.Index<0:0:2>) updates(%22 : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x3x62x62xf16, [@CMX_NN, 0]>, outputType = memref<1x3x62x62xf16, @DDR>>) -> !VPURegMapped.Index<0:1:1>
  %160 = VPUMI40XX.PlatformInfo -> <0:0:0>
  %161 = VPUMI40XX.ActShaveRt kernel("nnActEntry") -> !VPURegMapped.Index<0:0:0>
  %162 = VPURegMapped.Enqueue at(%19 : !VPURegMapped.Index<0:0:0>) (%153 -> %154 : <0:0:0> -> <0:0:1>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<ActKernelInvocation>}
  %163 = VPUMI40XX.MappedInference dmas((%155, %158) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) actKernelRanges((%151) : (!VPURegMapped.Index<0:0:0>)) actKernelInvocations((%153) : (!VPURegMapped.Index<0:0:0>)) barriers(%19 : !VPURegMapped.Index<0:0:0>) workItemTasks(%162 : !VPURegMapped.Index<0:0:0>) actShaveRt(%161 : !VPURegMapped.Index<0:0:0>) dmaHwpBase(%4 : memref<16xui32, [@CMX_NN, 0]>) dmaCount([[3, 2]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([[2, 0]]) actKernelInvocationsCount([[2, 0]]) mediaCount(0) barrierCount(4) workItemCount(1) finalBarrierId(3) -> !VPURegMapped.Index<0:0:0>
  return %arg1 : memref<1x1000x1x1xf16, @DDR>
  }
}

// CHECK-NOT: VPURegMapped.FetchTask
//CHECK: [[TB23:%.+]] = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:30>
//CHECK: [[TB24:%.+]] = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:31>
//CHECK: [[TB25:%.+]] = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:0:30>
//CHECK: [[TB26:%.+]] = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:0:31>

//CHECK: [[KR27:%.+]] = VPUMI40XX.ActKernelRange taskLocation([[TB25]] : !VPURegMapped.Index<0:0:30>)
//CHECK: [[KR28:%.+]] = VPUMI40XX.ActKernelRange taskLocation([[TB26]] : !VPURegMapped.Index<0:0:31>)
//CHECK: [[KI29:%.+]] = VPUMI40XX.ActKernelInvocation taskLocation([[TB23]] : !VPURegMapped.Index<0:0:30>)
//CHECK: [[KI30:%.+]] = VPUMI40XX.ActKernelInvocation {lastSecondaryTaskInExecutionGroup} taskLocation([[TB24]] : !VPURegMapped.Index<0:0:31>)

//CHECK: [[TR31:%.+]] = VPURegMapped.ViewTaskRange([[KR27]] -> [[KR28]] : <0:0:0> -> <0:0:1>) -> memref<2x40xui8>
//CHECK: [[TR32:%.+]] = VPURegMapped.ViewTaskRange([[TB25]] -> [[TB26]] : <0:0:30> -> <0:0:31>) -> memref<2x40xui8, [@CMX_NN, 0]>
//CHECK: [[TR33:%.+]] = VPURegMapped.ViewTaskRange([[KI29]] -> [[KI30]] : <0:0:0> -> <0:0:1>) -> memref<2x96xui8>
//CHECK: [[TR34:%.+]] = VPURegMapped.ViewTaskRange([[TB23]] -> [[TB24]]  : <0:0:30> -> <0:0:31>) -> memref<2x96xui8, [@CMX_NN, 0]>

//CHECK: [[DMA35:%.+]] = VPUMI40XX.NNDMA <{is_critical, is_out_of_order, port = 0 : i64}> inputs([[TR31]] : memref<2x40xui8>) outputs([[TR32]] : memref<2x40xui8, [@CMX_NN, 0]>)
//CHECK: VPUMI40XX.NNDMA <{is_critical, is_out_of_order, port = 0 : i64}> inputs([[TR33]] : memref<2x96xui8>) outputs([[TR34]] : memref<2x96xui8, [@CMX_NN, 0]>) previousDMA([[DMA35]] : !VPURegMapped.Index<0:0:0>)

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
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x115x224xf16, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x224x4x114x!qElemType, {order = #NWCH}, [@CMX_NN, 0]>
    %10 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x224x4x114x!qElemType, {order = #NWCH}, [@CMX_NN, 0]>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %12 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %13 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %14 = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %15 = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %16 = VPURT.DeclareBuffer <CMX_NN> [0] <200704> -> memref<1x64x28x56xf16, [@CMX_NN, 0]>
    %17 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %18 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %19 = VPURT.DeclareBuffer <CMX_NN> [0] <257600> -> memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %20 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x16x114x224x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>
    %21 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, isStartBarrier, producer_count = 1 : ui8, wlmPage = 0 : i64}> <0, 4> -> !VPURegMapped.Index<0:0:0>
    %22 = VPUMI40XX.ConfigureBarrier <{consumer_count = 2 : ui8, producer_count = 2 : ui8, wlmPage = 0 : i64}>(%21 : !VPURegMapped.Index<0:0:0>) <1, 5> -> !VPURegMapped.Index<0:0:1>
    %23 = VPUMI40XX.ConfigureBarrier <{consumer_count = 2 : ui8, producer_count = 1 : ui8, wlmPage = 0 : i64}>(%22 : !VPURegMapped.Index<0:0:1>) <2, 6> -> !VPURegMapped.Index<0:0:2>
    %24 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 2 : ui8, wlmPage = 0 : i64}>(%23 : !VPURegMapped.Index<0:0:2>) <3, 7> -> !VPURegMapped.Index<0:0:3>
    %25 = VPUMI40XX.ConfigureBarrier <{consumer_count = 2 : ui8, producer_count = 1 : ui8, wlmPage = 1 : i64}>(%24, %22 : !VPURegMapped.Index<0:0:3>, !VPURegMapped.Index<0:0:1>) <0, 8> previousSameId(%21 : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:4>
    %26 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8, wlmPage = 1 : i64}>(%25 : !VPURegMapped.Index<0:0:4>) <1, 9> previousSameId(%22 : !VPURegMapped.Index<0:0:1>) -> !VPURegMapped.Index<0:0:5>
    %27 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8, wlmPage = 1 : i64}>(%26 : !VPURegMapped.Index<0:0:5>) <2, 10> previousSameId(%23 : !VPURegMapped.Index<0:0:2>) -> !VPURegMapped.Index<0:0:6>
    %28 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8, wlmPage = 1 : i64}>(%27 : !VPURegMapped.Index<0:0:6>) <3, 11> previousSameId(%24 : !VPURegMapped.Index<0:0:3>) -> !VPURegMapped.Index<0:0:7>
    %29 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8, wlmPage = 2 : i64}>(%28, %25 : !VPURegMapped.Index<0:0:7>, !VPURegMapped.Index<0:0:4>) <0, 12> previousSameId(%25 : !VPURegMapped.Index<0:0:4>) -> !VPURegMapped.Index<0:0:8>
    %30 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8, wlmPage = 2 : i64}>(%29 : !VPURegMapped.Index<0:0:8>) <1, 13> previousSameId(%26 : !VPURegMapped.Index<0:0:5>) -> !VPURegMapped.Index<0:0:9>
    %31 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8, wlmPage = 2 : i64}>(%30 : !VPURegMapped.Index<0:0:9>) <2, -1> previousSameId(%27 : !VPURegMapped.Index<0:0:6>) -> !VPURegMapped.Index<0:0:10>
    %32 = VPUMI40XX.ConfigureBarrier <{consumer_count = 2 : ui8, producer_count = 1 : ui8, wlmPage = 2 : i64}>(%31 : !VPURegMapped.Index<0:0:10>) <3, -1> previousSameId(%28 : !VPURegMapped.Index<0:0:7>) -> !VPURegMapped.Index<0:0:11>
    %33 = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 2 : ui8, wlmPage = 3 : i64}>(%32 : !VPURegMapped.Index<0:0:11>) <0, -1> previousSameId(%29 : !VPURegMapped.Index<0:0:8>) -> !VPURegMapped.Index<0:0:12>
    %34 = VPUMI40XX.ConfigureBarrier <{consumer_count = 0 : ui8, isFinalBarrier, producer_count = 1 : ui8, wlmPage = 3 : i64}>(%33 : !VPURegMapped.Index<0:0:12>) <1, -1> previousSameId(%30 : !VPURegMapped.Index<0:0:9>) -> !VPURegMapped.Index<0:0:13>
    %35 = VPUMI40XX.DeclareTaskBuffer {offset = 0 : ui64} <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
    %36 = VPUMI40XX.DeclareTaskBuffer {offset = 352 : ui64} <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
    %37 = VPUMI40XX.DeclareTaskBuffer {offset = 704 : ui64} <DPUInvariant> -> !VPURegMapped.Index<0:0:2>
    %38 = VPUMI40XX.DeclareTaskBuffer {offset = 1056 : ui64} <DPUInvariant> -> !VPURegMapped.Index<0:0:3>
    %39 = VPUMI40XX.DeclareTaskBuffer {offset = 1408 : ui64} <DPUVariant> -> !VPURegMapped.Index<0:0:0>
    %40 = VPUMI40XX.DeclareTaskBuffer {offset = 1632 : ui64} <DPUVariant> -> !VPURegMapped.Index<0:0:1>
    %41 = VPUMI40XX.DeclareTaskBuffer {offset = 1856 : ui64} <DPUVariant> -> !VPURegMapped.Index<0:0:2>
    %42 = VPUMI40XX.DeclareTaskBuffer {offset = 2080 : ui64} <DPUVariant> -> !VPURegMapped.Index<0:0:3>
    %43 = VPUMI40XX.DeclareTaskBuffer {offset = 2304 : ui64} <DPUVariant> -> !VPURegMapped.Index<0:0:4>
    %44 = VPUMI40XX.DeclareTaskBuffer {offset = 2528 : ui64} <DPUVariant> -> !VPURegMapped.Index<0:0:5>
    %45 = VPUMI40XX.DeclareTaskBuffer {offset = 2752 : ui64} <DPUVariant> -> !VPURegMapped.Index<0:0:6>
    %46 = VPUMI40XX.DeclareTaskBuffer {offset = 2976 : ui64} <DPUVariant> -> !VPURegMapped.Index<0:0:7>
    %47 = VPUMI40XX.DPUInvariant <{clean_after = 0 : ui64, is_permute_quantize, is_superdense, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, wlmPage = 0 : i64}> taskLocation(%36 : !VPURegMapped.Index<0:0:1>) input(%18 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%17 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%9 : memref<1x224x4x114x!qElemType, {order = #NWCH}, [@CMX_NN, 0]>) waits(%24, %22 : !VPURegMapped.Index<0:0:3>, !VPURegMapped.Index<0:0:1>) updates(%25 : !VPURegMapped.Index<0:0:4>) enqueueBarrier(%21 : !VPURegMapped.Index<0:0:0>) -> <0:0:0> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %48 = VPUMI40XX.DPUVariant taskLocation(%39 : !VPURegMapped.Index<0:0:0>) calls(%47 : <0:0:0>) weights(%17 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [113, 2, 223], inEnd = [113, 2, 223], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 0 : i64}> -> <0:0:0>
    %49 = VPUMI40XX.DPUVariant taskLocation(%40 : !VPURegMapped.Index<0:0:1>) previousTask(%48 : !VPURegMapped.Index<0:0:0>) calls(%47 : <0:0:0>) weights(%17 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [113, 2, 223], inEnd = [113, 2, 223], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 0 : i64}> -> <0:0:1>
    %50 = VPUMI40XX.DPUVariant taskLocation(%41 : !VPURegMapped.Index<0:0:2>) previousTask(%49 : !VPURegMapped.Index<0:0:1>) calls(%47 : <0:0:0>) weights(%17 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [113, 2, 223], inEnd = [113, 2, 223], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 0 : i64}> -> <0:0:2>
    %51 = VPUMI40XX.DPUVariant taskLocation(%42 : !VPURegMapped.Index<0:0:3>) previousTask(%50 : !VPURegMapped.Index<0:0:2>) calls(%47 : <0:0:0>) weights(%17 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) {lastSecondaryTaskInExecutionGroup} <{end = [113, 2, 223], inEnd = [113, 2, 223], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 0 : i64}> -> <0:0:3>
    %52 = VPUMI40XX.DPUInvariant <{clean_after = 0 : ui64, is_permute_quantize, is_superdense, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, wlmPage = 1 : i64}> taskLocation(%38 : !VPURegMapped.Index<0:0:3>) previousTask(%47 : !VPURegMapped.Index<0:0:0>) input(%18 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%17 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%10 : memref<1x224x4x114x!qElemType, {order = #NWCH}, [@CMX_NN, 0]>) waits(%26 : !VPURegMapped.Index<0:0:5>) updates(%27 : !VPURegMapped.Index<0:0:6>) enqueueBarrier(%21 : !VPURegMapped.Index<0:0:0>) -> <0:0:1> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %53 = VPUMI40XX.DPUVariant taskLocation(%43 : !VPURegMapped.Index<0:0:4>) previousTask(%51 : !VPURegMapped.Index<0:0:3>) calls(%52 : <0:0:1>) weights(%17 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [113, 2, 223], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 1 : i64}> -> <0:0:4>
    %54 = VPUMI40XX.DPUVariant taskLocation(%44 : !VPURegMapped.Index<0:0:5>) previousTask(%53 : !VPURegMapped.Index<0:0:4>) calls(%52 : <0:0:1>) weights(%17 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [113, 2, 223], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 1 : i64}> -> <0:0:5>
    %55 = VPUMI40XX.DPUVariant taskLocation(%45 : !VPURegMapped.Index<0:0:6>) previousTask(%54 : !VPURegMapped.Index<0:0:5>) calls(%52 : <0:0:1>) weights(%17 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [113, 2, 223], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 1 : i64}> -> <0:0:6>
    %56 = VPUMI40XX.DPUVariant taskLocation(%46 : !VPURegMapped.Index<0:0:7>) previousTask(%55 : !VPURegMapped.Index<0:0:6>) calls(%52 : <0:0:1>) weights(%17 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) {lastSecondaryTaskInExecutionGroup} <{end = [113, 2, 223], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 1 : i64}> -> <0:0:7>
    %57 = VPUMI40XX.DPUInvariant <{clean_after = 0 : ui64, cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, wlmPage = 1 : i64}> taskLocation(%36 : !VPURegMapped.Index<0:0:1>) previousTask(%52 : !VPURegMapped.Index<0:0:1>) input(%20 : memref<1x16x114x224x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>) weights(%19 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%12 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) waits(%28, %25 : !VPURegMapped.Index<0:0:7>, !VPURegMapped.Index<0:0:4>) updates(%29 : !VPURegMapped.Index<0:0:8>) enqueueBarrier(%21 : !VPURegMapped.Index<0:0:0>) -> <0:0:2> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %58 = VPUMI40XX.DPUVariant taskLocation(%39 : !VPURegMapped.Index<0:0:0>) previousTask(%56 : !VPURegMapped.Index<0:0:7>) calls(%57 : <0:0:2>) weights(%19 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [111, 55, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 1 : i64}> -> <0:0:8>
    %59 = VPUMI40XX.DPUVariant taskLocation(%40 : !VPURegMapped.Index<0:0:1>) previousTask(%58 : !VPURegMapped.Index<0:0:8>) calls(%57 : <0:0:2>) weights(%19 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [111, 55, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 1 : i64}> -> <0:0:9>
    %60 = VPUMI40XX.DPUVariant taskLocation(%41 : !VPURegMapped.Index<0:0:2>) previousTask(%59 : !VPURegMapped.Index<0:0:9>) calls(%57 : <0:0:2>) weights(%19 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [111, 55, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 1 : i64}> -> <0:0:10>
    %61 = VPUMI40XX.DPUVariant taskLocation(%42 : !VPURegMapped.Index<0:0:3>) previousTask(%60 : !VPURegMapped.Index<0:0:10>) calls(%57 : <0:0:2>) weights(%19 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) {lastSecondaryTaskInExecutionGroup} <{end = [111, 55, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 1 : i64}> -> <0:0:11>
    %62 = VPUMI40XX.DPUInvariant <{clean_after = 0 : ui64, cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, wlmPage = 2 : i64}> taskLocation(%38 : !VPURegMapped.Index<0:0:3>) previousTask(%57 : !VPURegMapped.Index<0:0:2>) input(%20 : memref<1x16x114x224x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>) weights(%19 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%13 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) waits(%30 : !VPURegMapped.Index<0:0:9>) updates(%31 : !VPURegMapped.Index<0:0:10>) enqueueBarrier(%22 : !VPURegMapped.Index<0:0:1>) -> <0:0:3> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %63 = VPUMI40XX.DPUVariant taskLocation(%43 : !VPURegMapped.Index<0:0:4>) previousTask(%61 : !VPURegMapped.Index<0:0:11>) calls(%62 : <0:0:3>) weights(%19 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [111, 55, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:12>
    %64 = VPUMI40XX.DPUVariant taskLocation(%44 : !VPURegMapped.Index<0:0:5>) previousTask(%63 : !VPURegMapped.Index<0:0:12>) calls(%62 : <0:0:3>) weights(%19 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [111, 55, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:13>
    %65 = VPUMI40XX.DPUVariant taskLocation(%45 : !VPURegMapped.Index<0:0:6>) previousTask(%64 : !VPURegMapped.Index<0:0:13>) calls(%62 : <0:0:3>) weights(%19 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) <{end = [111, 55, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:14>
    %66 = VPUMI40XX.DPUVariant taskLocation(%46 : !VPURegMapped.Index<0:0:7>) previousTask(%65 : !VPURegMapped.Index<0:0:14>) calls(%62 : <0:0:3>) weights(%19 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) {lastSecondaryTaskInExecutionGroup} <{end = [111, 55, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:15>
    %67 = VPUMI40XX.DPUInvariant <{clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, start_after = 0 : ui64, wlmPage = 2 : i64}> taskLocation(%36 : !VPURegMapped.Index<0:0:1>) previousTask(%62 : !VPURegMapped.Index<0:0:3>) input(%11 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) outputs(%14 : memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) waits(%32 : !VPURegMapped.Index<0:0:11>) updates(%33 : !VPURegMapped.Index<0:0:12>) enqueueBarrier(%25 : !VPURegMapped.Index<0:0:4>) -> <0:0:4> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %68 = VPUMI40XX.DPUVariant taskLocation(%39 : !VPURegMapped.Index<0:0:0>) previousTask(%66 : !VPURegMapped.Index<0:0:15>) calls(%67 : <0:0:4>) <{end = [55, 27, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:16>
    %69 = VPUMI40XX.DPUVariant taskLocation(%40 : !VPURegMapped.Index<0:0:1>) previousTask(%68 : !VPURegMapped.Index<0:0:16>) calls(%67 : <0:0:4>) <{end = [55, 27, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:17>
    %70 = VPUMI40XX.DPUVariant taskLocation(%41 : !VPURegMapped.Index<0:0:2>) previousTask(%69 : !VPURegMapped.Index<0:0:17>) calls(%67 : <0:0:4>) <{end = [55, 27, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:18>
    %71 = VPUMI40XX.DPUVariant taskLocation(%42 : !VPURegMapped.Index<0:0:3>) previousTask(%70 : !VPURegMapped.Index<0:0:18>) calls(%67 : <0:0:4>) {lastSecondaryTaskInExecutionGroup} <{end = [55, 27, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:19>
    %72 = VPUMI40XX.DPUInvariant <{clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, start_after = 0 : ui64, wlmPage = 2 : i64}> taskLocation(%38 : !VPURegMapped.Index<0:0:3>) previousTask(%67 : !VPURegMapped.Index<0:0:4>) input(%11 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) outputs(%15 : memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) waits(%32 : !VPURegMapped.Index<0:0:11>) updates(%33 : !VPURegMapped.Index<0:0:12>) enqueueBarrier(%25 : !VPURegMapped.Index<0:0:4>) -> <0:0:5> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %73 = VPUMI40XX.DPUVariant taskLocation(%43 : !VPURegMapped.Index<0:0:4>) previousTask(%71 : !VPURegMapped.Index<0:0:19>) calls(%72 : <0:0:5>) <{end = [55, 27, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:20>
    %74 = VPUMI40XX.DPUVariant taskLocation(%44 : !VPURegMapped.Index<0:0:5>) previousTask(%73 : !VPURegMapped.Index<0:0:20>) calls(%72 : <0:0:5>) <{end = [55, 27, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:21>
    %75 = VPUMI40XX.DPUVariant taskLocation(%45 : !VPURegMapped.Index<0:0:6>) previousTask(%74 : !VPURegMapped.Index<0:0:21>) calls(%72 : <0:0:5>) <{end = [55, 27, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:22>
    %76 = VPUMI40XX.DPUVariant taskLocation(%46 : !VPURegMapped.Index<0:0:7>) previousTask(%75 : !VPURegMapped.Index<0:0:22>) calls(%72 : <0:0:5>) {lastSecondaryTaskInExecutionGroup} <{end = [55, 27, 63], inEnd = [111, 55, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, start = [0, 0, 0], wlmPage = 2 : i64}> -> <0:0:23>
    %77 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64, wlmPage = 0 : i64}> inputs(%3 : memref<0x0x0x0xi32, @DDR>) outputs(%4 : memref<0x0x0x0xi32, @DDR>) updates(%21 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %78 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64, wlmPage = 0 : i64}> inputs(%3 : memref<0x0x0x0xi32, @DDR>) outputs(%4 : memref<0x0x0x0xi32, @DDR>) previousDMA(%77 : !VPURegMapped.Index<0:0:0>) waits(%21 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %79 = VPURegMapped.FetchTask updates(%22 : !VPURegMapped.Index<0:0:1>) previousTask(%78 : !VPURegMapped.Index<0:0:1>) primary(%47 -> %47) secondary(%48 -> %51) (<0:0:0> -> <0:0:0> : !VPURegMapped.Index<0:0:0> -> !VPURegMapped.Index<0:0:3>) -> <0:0:2> {associated_execution_group_index = 0 : ui64, associated_task_type = #VPURegMapped.task_type<DPUInvariant>, associated_tile_index = 0 : ui64, wlmPage = -1 : i64}
    %80 = VPURegMapped.FetchTask updates(%22 : !VPURegMapped.Index<0:0:1>) previousTask(%79 : !VPURegMapped.Index<0:0:2>) primary(%52 -> %52) secondary(%53 -> %56) (<0:0:1> -> <0:0:1> : !VPURegMapped.Index<0:0:4> -> !VPURegMapped.Index<0:0:7>) -> <0:0:3> {associated_execution_group_index = 1 : ui64, associated_task_type = #VPURegMapped.task_type<DPUInvariant>, associated_tile_index = 0 : ui64, wlmPage = -1 : i64}
    %81 = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64, wlmPage = 0 : i64}> inputs(%5 : memref<0x0x0x0xi32, @DDR>) outputs(%6 : memref<0x0x0x0xi32, @DDR>) previousDMA(%80 : !VPURegMapped.Index<0:0:3>) waits(%22 : !VPURegMapped.Index<0:0:1>) updates(%23 : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:4>
    %82 = VPUMI40XX.NNDMA <{port = 0 : i64, wlmPage = 0 : i64}> inputs(%0 : memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%7 : memref<1x3x114x224xf16, [@CMX_NN, 0]>) previousDMA(%81 : !VPURegMapped.Index<0:0:4>) waits(%23 : !VPURegMapped.Index<0:0:2>) updates(%24 : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>, outputType = memref<1x3x114x224xf16, [@CMX_NN, 0]>>) -> !VPURegMapped.Index<0:0:5>
    %83 = VPUMI40XX.NNDMA <{port = 0 : i64, wlmPage = 0 : i64}> inputs(%1 : memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%8 : memref<1x3x115x224xf16, [@CMX_NN, 0]>) previousDMA(%82 : !VPURegMapped.Index<0:0:5>) waits(%23 : !VPURegMapped.Index<0:0:2>) updates(%24 : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>, outputType = memref<1x3x115x224xf16, [@CMX_NN, 0]>>) -> !VPURegMapped.Index<0:0:6>
    %84 = VPURegMapped.FetchTask waits(%25 : !VPURegMapped.Index<0:0:4>) updates(%26 : !VPURegMapped.Index<0:0:5>) previousTask(%83 : !VPURegMapped.Index<0:0:6>) primary(%57 -> %57) secondary(%58 -> %61) (<0:0:2> -> <0:0:2> : !VPURegMapped.Index<0:0:8> -> !VPURegMapped.Index<0:0:11>) -> <0:0:7> {associated_execution_group_index = 2 : ui64, associated_task_type = #VPURegMapped.task_type<DPUInvariant>, associated_tile_index = 0 : ui64, wlmPage = 1 : i64}
    %85 = VPURegMapped.FetchTask waits(%27 : !VPURegMapped.Index<0:0:6>) updates(%28 : !VPURegMapped.Index<0:0:7>) previousTask(%84 : !VPURegMapped.Index<0:0:7>) primary(%62 -> %62) secondary(%63 -> %66) (<0:0:3> -> <0:0:3> : !VPURegMapped.Index<0:0:12> -> !VPURegMapped.Index<0:0:15>) -> <0:0:8> {associated_execution_group_index = 3 : ui64, associated_task_type = #VPURegMapped.task_type<DPUInvariant>, associated_tile_index = 0 : ui64, wlmPage = 1 : i64}
    %86 = VPURegMapped.FetchTask waits(%29 : !VPURegMapped.Index<0:0:8>) updates(%30 : !VPURegMapped.Index<0:0:9>) previousTask(%85 : !VPURegMapped.Index<0:0:8>) primary(%67 -> %67) secondary(%68 -> %71) (<0:0:4> -> <0:0:4> : !VPURegMapped.Index<0:0:16> -> !VPURegMapped.Index<0:0:19>) enqueueBarrier(%22 : !VPURegMapped.Index<0:0:1>) -> <0:0:9> {associated_execution_group_index = 4 : ui64, associated_task_type = #VPURegMapped.task_type<DPUInvariant>, associated_tile_index = 0 : ui64, wlmPage = 2 : i64}
    %87 = VPURegMapped.FetchTask waits(%31 : !VPURegMapped.Index<0:0:10>) updates(%32 : !VPURegMapped.Index<0:0:11>) previousTask(%86 : !VPURegMapped.Index<0:0:9>) primary(%72 -> %72) secondary(%73 -> %76) (<0:0:5> -> <0:0:5> : !VPURegMapped.Index<0:0:20> -> !VPURegMapped.Index<0:0:23>) enqueueBarrier(%22 : !VPURegMapped.Index<0:0:1>) -> <0:0:10> {associated_execution_group_index = 5 : ui64, associated_task_type = #VPURegMapped.task_type<DPUInvariant>, associated_tile_index = 0 : ui64, wlmPage = 2 : i64}
    %88 = VPUMI40XX.NNDMA <{port = 0 : i64, wlmPage = 3 : i64}> inputs(%16 : memref<1x64x28x56xf16, [@CMX_NN, 0]>) outputs(%2 : memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>) waits(%33 : !VPURegMapped.Index<0:0:12>) updates(%34 : !VPURegMapped.Index<0:0:13>) enqueueBarrier(%25 : !VPURegMapped.Index<0:0:4>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x64x28x56xf16, [@CMX_NN, 0]>, outputType = memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>>) -> !VPURegMapped.Index<0:1:0>
    %89 = VPUMI40XX.MappedInference dmas((%77, %88) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) invariants(%47 : !VPURegMapped.Index<0:0:0>) variants(%48 : !VPURegMapped.Index<0:0:0>) barriers(%21 : !VPURegMapped.Index<0:0:0>) dmaCount([[11, 1], [0, 0]]) invariantCount([6, 0, 0, 0, 0, 0]) variantCount([24, 0, 0, 0, 0, 0]) actKernelRangesCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) actKernelInvocationsCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) mediaCount(0) barrierCount(14) finalBarrierId(13) -> !VPURegMapped.Index<0:0:0>
    VPUMI40XX.OpRanges types([#VPURegMapped.task_type<DMA>, #VPURegMapped.task_type<DPUInvariant>, #VPURegMapped.task_type<DPUVariant>, #VPURegMapped.task_type<DMA>]) begins(%77, %47, %48, %88 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>) ends(%87, %72, %76, %88 : !VPURegMapped.Index<0:0:10>, !VPURegMapped.Index<0:0:5>, !VPURegMapped.Index<0:0:23>, !VPURegMapped.Index<0:1:0>)
  }

  //CHECK: [[BAR_INIT:%.+]] = VPUMI40XX.ConfigureBarrier
  //CHECK: [[EB:%.+]] = VPUMI40XX.ConfigureBarrier <{consumer_count = 2 : ui8, producer_count = 2 : ui8, wlmPage = 0 : i64}>([[BAR_INIT]] : !VPURegMapped.Index<0:0:0>) <1, 5> -> !VPURegMapped.Index<0:0:1>
  // CHECK-NOT: VPURegMapped.FetchTask
  //CHECK: [[TB25:%.+]] = VPUMI40XX.DeclareTaskBuffer {offset = 1056 : ui64} <DPUInvariant> -> !VPURegMapped.Index<0:0:3>

  //CHECK: [[TB26:%.+]] = VPUMI40XX.DeclareTaskBuffer {offset = 2304 : ui64} <DPUVariant> -> !VPURegMapped.Index<0:0:4>
  //CHECK: [[TB27:%.+]] = VPUMI40XX.DeclareTaskBuffer {offset = 2528 : ui64} <DPUVariant> -> !VPURegMapped.Index<0:0:5>
  //CHECK: [[TB28:%.+]] = VPUMI40XX.DeclareTaskBuffer {offset = 2752 : ui64} <DPUVariant> -> !VPURegMapped.Index<0:0:6>
  //CHECK: [[TB29:%.+]] = VPUMI40XX.DeclareTaskBuffer {offset = 2976 : ui64} <DPUVariant> -> !VPURegMapped.Index<0:0:7>

  //CHECK: [[DI30:%.+]] = VPUMI40XX.DPUInvariant <{clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, start_after = 0 : ui64, wlmPage = 2 : i64}> taskLocation([[TB25]] : !VPURegMapped.Index<0:0:3>)

  //CHECK: [[DV31:%.+]] = VPUMI40XX.DPUVariant
  //CHECK-SAME: calls([[DI30]] : <0:0:5>)
  //CHECK: [[DV32:%.+]] = VPUMI40XX.DPUVariant
  //CHECK-SAME: calls([[DI30]] : <0:0:5>)
  //CHECK: [[DV33:%.+]] = VPUMI40XX.DPUVariant
  //CHECK-SAME: calls([[DI30]] : <0:0:5>)
  //CHECK: [[DV34:%.+]] = VPUMI40XX.DPUVariant
  //CHECK-SAME: calls([[DI30]] : <0:0:5>)


  //CHECK: [[TR31:%.+]] = VPURegMapped.ViewTaskRange([[DI30]] -> [[DI30]] : <0:0:5> -> <0:0:5>) -> memref<1x352xui8>
  //CHECK: [[TR32:%.+]] = VPURegMapped.ViewTaskRange([[TB25]] -> [[TB25]] : <0:0:3> -> <0:0:3>) -> memref<1x352xui8, [@CMX_NN, 0]>
  //CHECK: [[TR33:%.+]] = VPURegMapped.ViewTaskRange([[DV31]] -> [[DV34]] : <0:0:20> -> <0:0:23>) -> memref<4x224xui8>
  //CHECK: [[TR34:%.+]] = VPURegMapped.ViewTaskRange([[TB26]] -> [[TB29]]  : <0:0:4> -> <0:0:7>) -> memref<4x224xui8, [@CMX_NN, 0]>

  //CHECK: [[DMA35:%.+]] = VPUMI40XX.NNDMA <{is_critical, is_out_of_order, port = 0 : i64, wlmPage = 2 : i64}> inputs([[TR31]] : memref<1x352xui8>) outputs([[TR32]] : memref<1x352xui8, [@CMX_NN, 0]>)
  //CHECK-SAME enqueueBarrier([[EB]] : !VPURegMapped.Index<0:0:1>)
  //CHECK: VPUMI40XX.NNDMA <{is_critical, is_out_of_order, port = 0 : i64, wlmPage = 2 : i64}> inputs([[TR33]] : memref<4x224xui8>) outputs([[TR34]] : memref<4x224xui8, [@CMX_NN, 0]>) previousDMA([[DMA35]] : !VPURegMapped.Index<0:0:15>)
  //CHECK-SAME enqueueBarrier([[EB]] : !VPURegMapped.Index<0:0:1>)
}
