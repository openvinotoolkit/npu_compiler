//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --ungroup-execution-ops %s | FileCheck %s
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
    %17 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:2>
    %18 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:3>
    %19 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:4>
    %79 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
    %80 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
    %81 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:2>
    %82 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:3>
    %83 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:4>
    %startIndexes:2, %endIndexes:2 = "VPURegMapped.ExecutionGroup"(%11, %13) ({
      %215 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64} taskLocation(%15 : !VPURegMapped.Index<0:0:0>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%12 : !VPURegMapped.Index<0:0:2>) -> <0:0:0> PPE : {
        VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
      }
      %216 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} taskLocation(%16 : !VPURegMapped.Index<0:0:1>) previousTask(%215 : !VPURegMapped.Index<0:0:0>) input(%7 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) waits(%12 : !VPURegMapped.Index<0:0:2>) updates(%13 : !VPURegMapped.Index<0:0:3>) -> <0:0:1> PPE : {
        VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
      }
      %217 = VPUMI40XX.DPUVariant taskLocation(%79 : !VPURegMapped.Index<0:0:0>) calls(%215 : <0:0:0>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
      %218 = VPUMI40XX.DPUVariant taskLocation(%80 : !VPURegMapped.Index<0:0:1>) previousTask(%217 : !VPURegMapped.Index<0:0:0>) calls(%216 : <0:0:1>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
      "VPURegMapped.GroupYield"(%215, %217, %216, %218) {operandSegmentSizes = array<i32: 2, 2>} : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>) -> ()
    }) {operandSegmentSizes = array<i32: 0, 1, 1>, resultSegmentSizes = array<i32: 2, 2>, task_type = #VPURegMapped.task_type<DPUInvariant>} : (!VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:3>) -> (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>)
    %207 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %208 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %209 = VPURegMapped.FetchTask primary(%startIndexes#0 -> %endIndexes#0) secondary(%startIndexes#1 -> %endIndexes#1) (<0:0:0> -> <0:0:1> : !VPURegMapped.Index<0:0:0> -> !VPURegMapped.Index<0:0:1>) -> <0:0:0>
    %210 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 0 : i32, srcWidth = 0 : i32, srcStride = 0 : i32, srcPlaneStride = 0 : i32, dstWidth = 0 : i32, dstStride = 0 : i32, dstPlaneStride = 0 : i32>, port = 1 : i64} inputs(%207 : memref<1x1x1x1xi32, @DDR>) outputs(%208 : memref<1x1x1x1xi32, @DDR>) previousDMA(%209 : !VPURegMapped.Index<0:0:0>) updates(%10 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %211 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%0 : memref<1x16x16x16xf16, @DDR>) outputs(%2 : memref<1x16x16x16xf16, [@CMX_NN, 0]>) previousDMA(%210 : !VPURegMapped.Index<0:0:1>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%11 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %212 = VPUMI40XX.NNDMA {is_out_of_order, port = 1 : i64} inputs(%cst : memref<1x1x1x4864xui8>) outputs(%5 : memref<1x1x1x4864xui8, [@CMX_NN, 0]>) previousDMA(%211 : !VPURegMapped.Index<0:0:2>) updates(%12 : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>
    %213 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x16x14x14xf16, @DDR>) waits(%13 : !VPURegMapped.Index<0:0:3>) updates(%14 : !VPURegMapped.Index<0:0:4>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %214 = VPUMI40XX.MappedInference dmas((%209, %213) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) invariants(%startIndexes#0 : !VPURegMapped.Index<0:0:0>) variants(%startIndexes#1 : !VPURegMapped.Index<0:0:0>) barriers(%10 : !VPURegMapped.Index<0:0:0>) dmaCount([[4, 1]]) invariantCount([2]) variantCount([2]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(5) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x14x14xf16, @DDR>
  }
}

// CHECK-NOT: VPURegMapped.ExecutionGroup
// CHECK-NOT: VPURegMapped.GroupYield

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
  %startIndexes:2, %endIndexes:2 = "VPURegMapped.ExecutionGroup"(%20, %21) <{operandSegmentSizes = array<i32: 0, 1, 1>, resultSegmentSizes = array<i32: 2, 2>, task_type = #VPURegMapped.task_type<ActKernelRange>}> ({
    %159 = VPUMI40XX.ActKernelRange taskLocation(%117 : !VPURegMapped.Index<0:0:30>) kernel_text_index(%13 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%15 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%14 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
    %160 = VPUMI40XX.ActKernelRange taskLocation(%118 : !VPURegMapped.Index<0:0:31>) previousTask(%159 : !VPURegMapped.Index<0:0:0>) kernel_text_index(%13 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%16 : !VPURegMapped.Index<0:0:1>) kernel_entry_index(%14 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:1>
    %161 = VPUMI40XX.ActKernelInvocation taskLocation(%53 : !VPURegMapped.Index<0:0:30>) range_index(%159 : <0:0:0>) kernel_params(%17 : <0:0:0>) waits(%20 : !VPURegMapped.Index<0:0:1>) updates(%21 : !VPURegMapped.Index<0:0:2>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %162 = VPUMI40XX.ActKernelInvocation {lastSecondaryTaskInExecutionGroup} taskLocation(%54 : !VPURegMapped.Index<0:0:31>) previousTask(%161 : !VPURegMapped.Index<0:0:0>) range_index(%160 : <0:0:1>) kernel_params(%18 : <0:0:1>) waits(%20 : !VPURegMapped.Index<0:0:1>) updates(%21 : !VPURegMapped.Index<0:0:2>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
    "VPURegMapped.GroupYield"(%159, %161, %160, %162) <{operandSegmentSizes = array<i32: 2, 2>}> : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>) -> ()
  }) : (!VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:2>) -> (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>)
  %151 = VPURegMapped.FetchTask primary(%startIndexes#0 -> %endIndexes#0) secondary(%startIndexes#1 -> %endIndexes#1) (<0:0:0> -> <0:0:1> : !VPURegMapped.Index<0:0:0> -> !VPURegMapped.Index<0:0:1>) -> <0:0:0> {associated_execution_group_index = 0 : ui64, associated_task_type = #VPURegMapped.task_type<ActKernelRange>, associated_tile_index = 0 : ui64}
  %152 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64} inputs(%2 : memref<0x0x0x0xi32, @DDR>) outputs(%3 : memref<0x0x0x0xi32, @DDR>) previousDMA(%151 : !VPURegMapped.Index<0:0:0>) updates(%19 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
  %153 = VPUMI40XX.NNDMA {allow_different_in_out_shapes, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 3 : i64, len = 7688 : i64, srcWidth = 7688 : i64, srcStride = 2 : i64, srcPlaneStride = 7688 : i64, dstWidth = 2 : i64, dstStride = 6 : i64, dstPlaneStride = 2 : i64>, is_out_of_order, port = 0 : i64} inputs(%0 : memref<3x3844xf16, @DDR>) outputs(%5 : memref<3844x3xf16, [@CMX_NN, 0]>) previousDMA(%152 : !VPURegMapped.Index<0:0:1>) waits(%19 : !VPURegMapped.Index<0:0:0>) updates(%20 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
  %154 = VPUMI40XX.NNDMA {allow_different_in_out_shapes, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 3844 : i64, len = 6 : i64, srcWidth = 6 : i64, srcStride = 2 : i64, srcPlaneStride = 6 : i64, dstWidth = 2 : i64, dstStride = 7688 : i64, dstPlaneStride = 2 : i64>, port = 0 : i64} inputs(%6 : memref<3844x3xf16, [@CMX_NN, 0]>) outputs(%8 : memref<3x3844xf16, [@CMX_NN, 0]>) waits(%21 : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
  %155 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%7 : memref<1x3x62x62xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x3x62x62xf16, @DDR>) previousDMA(%154 : !VPURegMapped.Index<0:1:0>) waits(%21 : !VPURegMapped.Index<0:0:2>) updates(%22 : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x3x62x62xf16, [@CMX_NN, 0]>, outputType = memref<1x3x62x62xf16, @DDR>>) -> !VPURegMapped.Index<0:1:1>
  %156 = VPUMI40XX.PlatformInfo -> <0:0:0>
  %157 = VPUMI40XX.ActShaveRt kernel("nnActEntry") -> !VPURegMapped.Index<0:0:0>
  %158 = VPUMI40XX.MappedInference dmas((%151, %154) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) actKernelRanges((%startIndexes#0) : (!VPURegMapped.Index<0:0:0>)) actKernelInvocations((%startIndexes#1) : (!VPURegMapped.Index<0:0:0>)) barriers(%19 : !VPURegMapped.Index<0:0:0>) actShaveRt(%157 : !VPURegMapped.Index<0:0:0>) dmaHwpBase(%4 : memref<16xui32, [@CMX_NN, 0]>) dmaCount([[3, 2]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([[2, 0]]) actKernelInvocationsCount([[2, 0]]) mediaCount(0) barrierCount(4) -> !VPURegMapped.Index<0:0:0>
  return %arg1 : memref<1x1000x1x1xf16, @DDR>
  }
}

// CHECK-NOT: VPURegMapped.ExecutionGroup
// CHECK-NOT: VPURegMapped.GroupYield
