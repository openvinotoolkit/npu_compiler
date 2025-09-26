//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --split-enqueue-ops %s | FileCheck %s
// REQUIRES: arch-NPU40XX


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @Convolution attributes {config.arch = #config.arch_kind<NPU40XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
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
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x14x14xf16, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <16896> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <17152> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %10 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <4, -1> -> !VPURegMapped.Index<0:0:0>
    %14 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, isFinalBarrier, producer_count = 1 : ui8}(%10 : !VPURegMapped.Index<0:0:0>) <3, -1> -> !VPURegMapped.Index<0:0:1>
    %15 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
    %16 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
    %17 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
    %18 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
    %19 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%15 : !VPURegMapped.Index<0:0:0>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%14 : !VPURegMapped.Index<0:0:1>) -> <0:0:0> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %20 = VPUMI40XX.DPUInvariant {clean_after = 3 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 4 : ui64} taskLocation(%16 : !VPURegMapped.Index<0:0:1>) previousTask(%19 : !VPURegMapped.Index<0:0:0>) input(%7 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%14 : !VPURegMapped.Index<0:0:1>) -> <0:0:1> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %21 = VPUMI40XX.DPUVariant taskLocation(%17 : !VPURegMapped.Index<0:0:0>) calls(%19 : <0:0:0>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], lastSecondaryTaskInExecutionGroup, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
    %22 = VPUMI40XX.DPUVariant taskLocation(%18 : !VPURegMapped.Index<0:0:1>) previousTask(%21 : !VPURegMapped.Index<0:0:0>) calls(%20 : <0:0:1>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
    %25 = VPURegMapped.ViewTaskRange(%19 -> %20 : <0:0:0> -> <0:0:1>) -> memref<2x352xui8>
    %26 = VPURegMapped.ViewTaskRange(%15 -> %16 : <0:0:0> -> <0:0:1>) -> memref<2x352xui8, [@CMX_NN, 0]>
    %35 = VPURegMapped.Enqueue at(%10 : !VPURegMapped.Index<0:0:0>) (%21 -> %22 : <0:0:0> -> <0:0:1>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<DPUVariant>}
    %37 = VPUMI40XX.MappedInference invariants(%19 : !VPURegMapped.Index<0:0:0>) variants(%21 : !VPURegMapped.Index<0:0:0>) barriers(%10 : !VPURegMapped.Index<0:0:0>) workItemTasks(%35 : !VPURegMapped.Index<0:0:0>) dmaCount([[0, 0]]) invariantCount([2]) variantCount([2]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(5) workItemCount(1) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x14x14xf16, @DDR>
  }
}

//CHECK: [[VAL10:%.+]] = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
//CHECK: [[VAL11:%.+]] = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
//CHECK: [[VAL12:%.+]] = VPUMI40XX.DPUInvariant
//CHECK: [[VAL13:%.+]] = VPUMI40XX.DPUInvariant
//CHECK: [[VAL14:%.+]] = VPUMI40XX.DPUVariant taskLocation([[VAL10]] : !VPURegMapped.Index<0:0:0>) calls([[VAL12]] : <0:0:0>)
//CHECK: [[VAL15:%.+]] = VPUMI40XX.DPUVariant taskLocation([[VAL11]] : !VPURegMapped.Index<0:0:1>) previousTask([[VAL14]] : !VPURegMapped.Index<0:0:0>) calls([[VAL13]] : <0:0:1>)
//CHECK: [[VAL16:%.+]] = VPURegMapped.Enqueue
//CHECK-SAME: ([[VAL14]] -> [[VAL14]] : <0:0:0> -> <0:0:0>)
//CHECK: [[VAL17:%.+]] = VPURegMapped.Enqueue
//CHECK-SAME: ([[VAL15]] -> [[VAL15]] : <0:0:1> -> <0:0:1>)
//CHECK: workItemCount(2)

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @Convolution attributes {config.arch = #config.arch_kind<NPU40XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
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
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x14x14xf16, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <16896> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <17152> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %10 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <4, -1> -> !VPURegMapped.Index<0:0:0>
    %14 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, isFinalBarrier, producer_count = 1 : ui8}(%10 : !VPURegMapped.Index<0:0:0>) <3, -1> -> !VPURegMapped.Index<0:0:1>
    %15 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
    %16 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
    %17 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
    %18 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
    %19 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%15 : !VPURegMapped.Index<0:0:0>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%14 : !VPURegMapped.Index<0:0:1>) -> <0:0:0> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %20 = VPUMI40XX.DPUInvariant {clean_after = 3 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 4 : ui64} taskLocation(%16 : !VPURegMapped.Index<0:0:1>) previousTask(%19 : !VPURegMapped.Index<0:0:0>) input(%7 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%14 : !VPURegMapped.Index<0:0:1>) -> <0:0:1> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %21 = VPUMI40XX.DPUVariant taskLocation(%17 : !VPURegMapped.Index<0:0:0>) calls(%19 : <0:0:0>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
    %22 = VPUMI40XX.DPUVariant taskLocation(%18 : !VPURegMapped.Index<0:0:1>) previousTask(%21 : !VPURegMapped.Index<0:0:0>) calls(%20 : <0:0:1>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
    %23 = VPUMI40XX.DPUVariant taskLocation(%17 : !VPURegMapped.Index<0:0:0>) previousTask(%22 : !VPURegMapped.Index<0:0:1>) calls(%19 : <0:0:0>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], lastSecondaryTaskInExecutionGroup, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:2>
    %24 = VPUMI40XX.DPUVariant taskLocation(%18 : !VPURegMapped.Index<0:0:1>) previousTask(%23 : !VPURegMapped.Index<0:0:2>) calls(%20 : <0:0:1>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:3>
    %25 = VPURegMapped.ViewTaskRange(%19 -> %20 : <0:0:0> -> <0:0:1>) -> memref<2x352xui8>
    %26 = VPURegMapped.ViewTaskRange(%15 -> %16 : <0:0:0> -> <0:0:1>) -> memref<2x352xui8, [@CMX_NN, 0]>
    %35 = VPURegMapped.Enqueue at(%10 : !VPURegMapped.Index<0:0:0>) (%21 -> %22 : <0:0:0> -> <0:0:1>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<DPUVariant>}
    %36 = VPURegMapped.Enqueue previousTaskIdx(%35 : !VPURegMapped.Index<0:0:0>) at(%10 : !VPURegMapped.Index<0:0:0>) (%23 -> %24 : <0:0:2> -> <0:0:3>) -> !VPURegMapped.Index<0:0:1> {taskType = #VPURegMapped.task_type<DPUVariant>}
    %37 = VPUMI40XX.MappedInference invariants(%19 : !VPURegMapped.Index<0:0:0>) variants(%21 : !VPURegMapped.Index<0:0:0>) barriers(%10 : !VPURegMapped.Index<0:0:0>) workItemTasks(%35 : !VPURegMapped.Index<0:0:0>) dmaCount([[0, 0]]) invariantCount([2]) variantCount([2]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(5) workItemCount(2) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x14x14xf16, @DDR>
  }
}

//CHECK: [[VAL10:%.+]] = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
//CHECK: [[VAL11:%.+]] = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
//CHECK: [[VAL12:%.+]] = VPUMI40XX.DPUInvariant
//CHECK: [[VAL13:%.+]] = VPUMI40XX.DPUInvariant
//CHECK: [[VAL14:%.+]] = VPUMI40XX.DPUVariant taskLocation([[VAL10]] : !VPURegMapped.Index<0:0:0>) calls([[VAL12]] : <0:0:0>)
//CHECK: [[VAL15:%.+]] = VPUMI40XX.DPUVariant taskLocation([[VAL11]] : !VPURegMapped.Index<0:0:1>) previousTask([[VAL14]] : !VPURegMapped.Index<0:0:0>) calls([[VAL13]] : <0:0:1>)
//CHECK: [[VAL16:%.+]] = VPUMI40XX.DPUVariant taskLocation([[VAL10]] : !VPURegMapped.Index<0:0:0>) previousTask([[VAL15]] : !VPURegMapped.Index<0:0:1>) calls([[VAL12]] : <0:0:0>)
//CHECK: [[VAL17:%.+]] = VPUMI40XX.DPUVariant taskLocation([[VAL11]] : !VPURegMapped.Index<0:0:1>) previousTask([[VAL16]] : !VPURegMapped.Index<0:0:2>) calls([[VAL13]] : <0:0:1>)
//CHECK: [[VAL18:%.+]] = VPURegMapped.Enqueue
//CHECK-SAME: ([[VAL14]] -> [[VAL15]] : <0:0:0> -> <0:0:1>)
//CHECK: [[VAL19:%.+]] = VPURegMapped.Enqueue
//CHECK-SAME: ([[VAL16]] -> [[VAL16]] : <0:0:2> -> <0:0:2>)
//CHECK: [[VAL20:%.+]] = VPURegMapped.Enqueue
//CHECK-SAME: ([[VAL17]] -> [[VAL17]] : <0:0:3> -> <0:0:3>)
//CHECK: workItemCount(3)

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @ConvolutionMulipleGroupsInSingleEnq attributes {config.arch = #config.arch_kind<NPU40XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
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
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x14x14xf16, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <16896> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <17152> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %10 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <4, -1> -> !VPURegMapped.Index<0:0:0>
    %14 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, isFinalBarrier, producer_count = 1 : ui8}(%10 : !VPURegMapped.Index<0:0:0>) <3, -1> -> !VPURegMapped.Index<0:0:1>

    %15 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
    %16 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
    %17 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:2>
    %18 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:3>

    %19 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
    %20 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
    %21 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:2>
    %22 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:3>

    %23 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%15 : !VPURegMapped.Index<0:0:0>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%14 : !VPURegMapped.Index<0:0:1>) -> <0:0:0> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %24 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%16 : !VPURegMapped.Index<0:0:1>) previousTask(%23 : !VPURegMapped.Index<0:0:0>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%14 : !VPURegMapped.Index<0:0:1>) -> <0:0:1> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %25 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%17 : !VPURegMapped.Index<0:0:2>) previousTask(%24 : !VPURegMapped.Index<0:0:1>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%14 : !VPURegMapped.Index<0:0:1>) -> <0:0:2> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %26 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%18 : !VPURegMapped.Index<0:0:3>) previousTask(%25 : !VPURegMapped.Index<0:0:2>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%14 : !VPURegMapped.Index<0:0:1>) -> <0:0:3> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }

    %27 = VPUMI40XX.DPUVariant taskLocation(%19 : !VPURegMapped.Index<0:0:0>) calls(%23 : <0:0:0>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], lastSecondaryTaskInExecutionGroup, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
    %28 = VPUMI40XX.DPUVariant taskLocation(%20 : !VPURegMapped.Index<0:0:1>) previousTask(%27 : !VPURegMapped.Index<0:0:0>) calls(%24 : <0:0:1>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], lastSecondaryTaskInExecutionGroup, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
    %29 = VPUMI40XX.DPUVariant taskLocation(%21 : !VPURegMapped.Index<0:0:2>) previousTask(%28 : !VPURegMapped.Index<0:0:1>) calls(%25 : <0:0:2>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], lastSecondaryTaskInExecutionGroup, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:2>
    %30 = VPUMI40XX.DPUVariant taskLocation(%22 : !VPURegMapped.Index<0:0:3>) previousTask(%29 : !VPURegMapped.Index<0:0:2>) calls(%26 : <0:0:3>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], lastSecondaryTaskInExecutionGroup, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:3>

    %31 = VPURegMapped.Enqueue at(%10 : !VPURegMapped.Index<0:0:0>) (%27 -> %30 : <0:0:0> -> <0:0:3>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<DPUVariant>}

    %32 = VPUMI40XX.MappedInference invariants(%23 : !VPURegMapped.Index<0:0:0>) variants(%27 : !VPURegMapped.Index<0:0:0>) barriers(%10 : !VPURegMapped.Index<0:0:0>) workItemTasks(%31 : !VPURegMapped.Index<0:0:0>) dmaCount([[0, 0]]) invariantCount([4]) variantCount([4]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(5) workItemCount(1) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x14x14xf16, @DDR>
  }
}

//CHECK: [[VAR_BUF0:%.+]] = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
//CHECK: [[VAR_BUF1:%.+]] = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
//CHECK: [[VAR_BUF2:%.+]] = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:2>
//CHECK: [[VAR_BUF3:%.+]] = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:3>

//CHECK: [[INV0:%.+]] = VPUMI40XX.DPUInvariant
//CHECK: [[INV1:%.+]] = VPUMI40XX.DPUInvariant
//CHECK: [[INV2:%.+]] = VPUMI40XX.DPUInvariant
//CHECK: [[INV3:%.+]] = VPUMI40XX.DPUInvariant

//CHECK: [[VAR0:%.+]] = VPUMI40XX.DPUVariant taskLocation([[VAR_BUF0]] : !VPURegMapped.Index<0:0:0>) calls([[INV0]] : <0:0:0>)
//CHECK: [[VAR1:%.+]] = VPUMI40XX.DPUVariant taskLocation([[VAR_BUF1]] : !VPURegMapped.Index<0:0:1>) previousTask([[VAR0]] : !VPURegMapped.Index<0:0:0>) calls([[INV1]] : <0:0:1>)
//CHECK: [[VAR2:%.+]] = VPUMI40XX.DPUVariant taskLocation([[VAR_BUF2]] : !VPURegMapped.Index<0:0:2>) previousTask([[VAR1]] : !VPURegMapped.Index<0:0:1>) calls([[INV2]] : <0:0:2>)
//CHECK: [[VAR3:%.+]] = VPUMI40XX.DPUVariant taskLocation([[VAR_BUF3]] : !VPURegMapped.Index<0:0:3>) previousTask([[VAR2]] : !VPURegMapped.Index<0:0:2>) calls([[INV3]] : <0:0:3>)

//CHECK: [[ENQ0:%.+]] = VPURegMapped.Enqueue
//CHECK-SAME: ([[VAR0]] -> [[VAR0]] : <0:0:0> -> <0:0:0>)
//CHECK: [[ENQ1:%.+]] = VPURegMapped.Enqueue
//CHECK-SAME: ([[VAR1]] -> [[VAR1]] : <0:0:1> -> <0:0:1>)
//CHECK: [[ENQ2:%.+]] = VPURegMapped.Enqueue
//CHECK-SAME: ([[VAR2]] -> [[VAR2]] : <0:0:2> -> <0:0:2>)
//CHECK: [[ENQ3:%.+]] = VPURegMapped.Enqueue
//CHECK-SAME: ([[VAR3]] -> [[VAR3]] : <0:0:3> -> <0:0:3>)
//CHECK: workItemTasks([[ENQ0]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: workItemCount(4)
