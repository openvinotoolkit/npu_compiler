//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --unroll-fetch-ops %s | FileCheck %s
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
    %19 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%15 : !VPURegMapped.Index<0:0:0>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%12 : !VPURegMapped.Index<0:0:2>) -> <0:0:0> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %20 = VPUMI40XX.DPUInvariant {clean_after = 3 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 4 : ui64} taskLocation(%16 : !VPURegMapped.Index<0:0:1>) previousTask(%19 : !VPURegMapped.Index<0:0:0>) input(%7 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) waits(%12 : !VPURegMapped.Index<0:0:2>) updates(%13 : !VPURegMapped.Index<0:0:3>) -> <0:0:1> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %21 = VPUMI40XX.DPUVariant taskLocation(%17 : !VPURegMapped.Index<0:0:0>) calls(%19 : <0:0:0>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
    %22 = VPUMI40XX.DPUVariant taskLocation(%18 : !VPURegMapped.Index<0:0:1>) previousTask(%21 : !VPURegMapped.Index<0:0:0>) calls(%20 : <0:0:1>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
    %23 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %24 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %25 = VPURegMapped.FetchTask primary(%19 -> %20) secondary(%21 -> %22) (<0:0:0> -> <0:0:1> : !VPURegMapped.Index<0:0:0> -> !VPURegMapped.Index<0:0:1>) -> <0:0:0>
    %26 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 0 : i32, srcWidth = 0 : i32, srcStride = 0 : i32, srcPlaneStride = 0 : i32, dstWidth = 0 : i32, dstStride = 0 : i32, dstPlaneStride = 0 : i32>, port = 1 : i64} inputs(%23 : memref<1x1x1x1xi32, @DDR>) outputs(%24 : memref<1x1x1x1xi32, @DDR>) previousDMA(%25 : !VPURegMapped.Index<0:0:0>) updates(%10 : !VPURegMapped.Index<0:0:0>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %27 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%0 : memref<1x16x16x16xf16, @DDR>) outputs(%2 : memref<1x16x16x16xf16, [@CMX_NN, 0]>) previousDMA(%26 : !VPURegMapped.Index<0:0:1>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%11 : !VPURegMapped.Index<0:0:1>) start_after(2) clean_after(1) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %28 = VPUMI40XX.NNDMA {is_out_of_order, port = 1 : i64} inputs(%cst : memref<1x1x1x4864xui8>) outputs(%5 : memref<1x1x1x4864xui8, [@CMX_NN, 0]>) previousDMA(%27 : !VPURegMapped.Index<0:0:2>) updates(%12 : !VPURegMapped.Index<0:0:2>) start_after(3) clean_after(2) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>
    %29 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x16x14x14xf16, @DDR>) waits(%13 : !VPURegMapped.Index<0:0:3>) updates(%14 : !VPURegMapped.Index<0:0:4>) start_after(5) clean_after(4) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %30 = VPURegMapped.Enqueue at(%10 : !VPURegMapped.Index<0:0:0>) (%21 -> %21 : <0:0:0> -> <0:0:0>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<DPUVariant>}
    %31 = VPURegMapped.Enqueue previousTaskIdx(%30 : !VPURegMapped.Index<0:0:0>) at(%10 : !VPURegMapped.Index<0:0:0>) (%22 -> %22 : <0:0:1> -> <0:0:1>) -> !VPURegMapped.Index<0:0:1> {taskType = #VPURegMapped.task_type<DPUVariant>}
    %32 = VPUMI40XX.MappedInference dmas((%25, %29) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) invariants(%19 : !VPURegMapped.Index<0:0:0>) variants(%21 : !VPURegMapped.Index<0:0:0>) barriers(%10 : !VPURegMapped.Index<0:0:0>) workItemTasks(%30 : !VPURegMapped.Index<0:0:0>) dmaCount([[4, 1]]) invariantCount([2]) variantCount([2]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(5) workItemCount(2) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x14x14xf16, @DDR>
  }
}

// CHECK-NOT: VPURegMapped.FetchTask
//CHECK: [[VAL15:%.*]] = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
//CHECK: [[VAL16:%.*]] = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
//CHECK: [[VAL17:%.*]] = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
//CHECK: [[VAL18:%.*]] = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
//CHECK: [[VAL19:%.*]] = VPUMI40XX.DPUInvariant
//CHECK: [[VAL20:%.*]] = VPUMI40XX.DPUInvariant
//CHECK: [[VAL21:%.*]] = VPUMI40XX.DPUVariant taskLocation([[VAL17]] : !VPURegMapped.Index<0:0:0>) calls([[VAL19]] : <0:0:0>)
//CHECK: [[VAL22:%.*]] = VPUMI40XX.DPUVariant taskLocation([[VAL18]] : !VPURegMapped.Index<0:0:1>) previousTask([[VAL21]] : !VPURegMapped.Index<0:0:0>) calls([[VAL20]] : <0:0:1>)
//CHECK: VPURegMapped.ViewTaskRange([[VAL19]] -> [[VAL20]] : <0:0:0> -> <0:0:1>) -> memref<2x{{.*}}xui8>
//CHECK: VPURegMapped.ViewTaskRange([[VAL15]] -> [[VAL16]] : <0:0:0> -> <0:0:1>) -> memref<2x{{.*}}xui8, [@CMX_NN, 0]>
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
  %17 = VPUMI40XX.KernelParams inputs(%9 : memref<1x3x32x62xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [11532, 1, 186, 3]}, [@CMX_NN, 0]>) outputs(%10 : memref<1x3x32x62xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [11532, 1, 186, 3]}, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:0:0>
  %18 = VPUMI40XX.KernelParams inputs(%11 : memref<1x3x30x62xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [11532, 1, 186, 3]}, [@CMX_NN, 0]>) outputs(%12 : memref<1x3x30x62xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [11532, 1, 186, 3]}, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([]) -> !VPURegMapped.Index<0:0:1>
  %19 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, isStartBarrier, producer_count = 1 : ui8} <0, -1> -> !VPURegMapped.Index<0:0:0>
  %20 = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 1 : ui8}(%19 : !VPURegMapped.Index<0:0:0>) <1, -1> -> !VPURegMapped.Index<0:0:1>
  %21 = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}(%20 : !VPURegMapped.Index<0:0:1>) <2, -1> -> !VPURegMapped.Index<0:0:2>
  %22 = VPUMI40XX.ConfigureBarrier {consumer_count = 0 : ui8, isFinalBarrier, producer_count = 1 : ui8}(%21 : !VPURegMapped.Index<0:0:2>) <3, -1> -> !VPURegMapped.Index<0:0:3>
  %53 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:30>
  %54 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:31>
  %117 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:0:30>
  %118 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:0:31>
  %151 = VPUMI40XX.ActKernelRange taskLocation(%117 : !VPURegMapped.Index<0:0:30>) kernel_text_index(%13 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%15 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%14 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
  %152 = VPUMI40XX.ActKernelRange taskLocation(%118 : !VPURegMapped.Index<0:0:31>) previousTask(%151 : !VPURegMapped.Index<0:0:0>) kernel_text_index(%13 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%16 : !VPURegMapped.Index<0:0:1>) kernel_entry_index(%14 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:1>
  %153 = VPUMI40XX.ActKernelInvocation taskLocation(%53 : !VPURegMapped.Index<0:0:30>) range_index(%151 : <0:0:0>) kernel_params(%17 : <0:0:0>) waits(%20 : !VPURegMapped.Index<0:0:1>) updates(%21 : !VPURegMapped.Index<0:0:2>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
  %154 = VPUMI40XX.ActKernelInvocation {lastSecondaryTaskInExecutionGroup} taskLocation(%54 : !VPURegMapped.Index<0:0:31>) previousTask(%153 : !VPURegMapped.Index<0:0:0>) range_index(%152 : <0:0:1>) kernel_params(%18 : <0:0:1>) waits(%20 : !VPURegMapped.Index<0:0:1>) updates(%21 : !VPURegMapped.Index<0:0:2>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
  %155 = VPURegMapped.FetchTask primary(%151 -> %152) secondary(%153 -> %154) (<0:0:0> -> <0:0:1> : !VPURegMapped.Index<0:0:0> -> !VPURegMapped.Index<0:0:1>) -> <0:0:0> {associated_execution_group_index = 0 : ui64, associated_task_type = #VPURegMapped.task_type<ActKernelRange>, associated_tile_index = 0 : ui64}
  %156 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64} inputs(%2 : memref<0x0x0x0xi32, @DDR>) outputs(%3 : memref<0x0x0x0xi32, @DDR>) previousDMA(%155 : !VPURegMapped.Index<0:0:0>) updates(%19 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
  %157 = VPUMI40XX.NNDMA {allow_different_in_out_shapes, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 3 : i64, len = 7688 : i64, srcWidth = 7688 : i64, srcStride = 2 : i64, srcPlaneStride = 7688 : i64, dstWidth = 2 : i64, dstStride = 6 : i64, dstPlaneStride = 2 : i64>, is_out_of_order, port = 0 : i64} inputs(%0 : memref<3x3844xf16, @DDR>) outputs(%5 : memref<3844x3xf16, [@CMX_NN, 0]>) previousDMA(%156 : !VPURegMapped.Index<0:0:1>) waits(%19 : !VPURegMapped.Index<0:0:0>) updates(%20 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
  %158 = VPUMI40XX.NNDMA {allow_different_in_out_shapes, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 3844 : i64, len = 6 : i64, srcWidth = 6 : i64, srcStride = 2 : i64, srcPlaneStride = 6 : i64, dstWidth = 2 : i64, dstStride = 7688 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64} inputs(%6 : memref<3844x3xf16, [@CMX_NN, 0]>) outputs(%8 : memref<3x3844xf16, [@CMX_NN, 0]>) waits(%21 : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
  %159 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%7 : memref<1x3x62x62xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x3x62x62xf16, @DDR>) previousDMA(%158 : !VPURegMapped.Index<0:1:0>) waits(%21 : !VPURegMapped.Index<0:0:2>) updates(%22 : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x3x62x62xf16, [@CMX_NN, 0]>, outputType = memref<1x3x62x62xf16, @DDR>>) -> !VPURegMapped.Index<0:1:1>
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

//CHECK: [[DMA35:%.+]] = VPUMI40XX.NNDMA {is_critical, is_out_of_order, port = 0 : i64} inputs([[TR31]] : memref<2x40xui8>) outputs([[TR32]] : memref<2x40xui8, [@CMX_NN, 0]>)
//CHECK: VPUMI40XX.NNDMA {is_critical, is_out_of_order, port = 0 : i64} inputs([[TR33]] : memref<2x96xui8>) outputs([[TR34]] : memref<2x96xui8, [@CMX_NN, 0]>) previousDMA([[DMA35]] : !VPURegMapped.Index<0:0:0>)
