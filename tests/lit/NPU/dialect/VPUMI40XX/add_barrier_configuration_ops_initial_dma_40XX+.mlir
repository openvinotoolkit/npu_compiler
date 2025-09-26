//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --add-barrier-configuration-ops="workload-management-barrier-programming-mode=INITIAL_BARRIER_DMAS_SCHEDULED" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @Convolution attributes {config.compilationMode = #config.compilation_mode<DefaultHW>, config.revisionID = #config.revision_id<REVISION_NONE>} {
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
    %startIndexes:2, %endIndexes:2 = "VPURegMapped.ExecutionGroup"(%11, %13) <{operandSegmentSizes = array<i32: 0, 1, 1>, resultSegmentSizes = array<i32: 2, 2>, task_type = #VPURegMapped.task_type<DPUInvariant>}> ({
      %23 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64} input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%12 : !VPURegMapped.Index<0:0:2>) -> <0:0:0> PPE : {
        VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
      }
      %24 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} previousTask(%23 : !VPURegMapped.Index<0:0:0>) input(%7 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) waits(%12 : !VPURegMapped.Index<0:0:2>) updates(%13 : !VPURegMapped.Index<0:0:3>) -> <0:0:1> PPE : {
        VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
      }
      %25 = VPUMI40XX.DPUVariant calls(%23 : <0:0:0>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
      %26 = VPUMI40XX.DPUVariant previousTask(%25 : !VPURegMapped.Index<0:0:0>) calls(%24 : <0:0:1>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
      "VPURegMapped.GroupYield"(%23, %25, %24, %26) <{operandSegmentSizes = array<i32: 2, 2>}> : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>) -> ()
    }) : (!VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:3>) -> (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>)
    %15 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %16 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %17 = VPURegMapped.FetchTask primary(%startIndexes#0 -> %endIndexes#0) secondary(%startIndexes#1 -> %endIndexes#1) (<0:0:0> -> <0:0:1> : !VPURegMapped.Index<0:0:0> -> !VPURegMapped.Index<0:0:1>) -> <0:0:0> {associated_execution_group_index = 0 : ui64, associated_task_type = #VPURegMapped.task_type<DPUInvariant>, associated_tile_index = 0 : ui64}
    %18 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 0 : i32, srcWidth = 0 : i32, srcStride = 0 : i32, srcPlaneStride = 0 : i32, dstWidth = 0 : i32, dstStride = 0 : i32, dstPlaneStride = 0 : i32>, port = 1 : i64} inputs(%15 : memref<1x1x1x1xi32, @DDR>) outputs(%16 : memref<1x1x1x1xi32, @DDR>) previousDMA(%17 : !VPURegMapped.Index<0:0:0>) updates(%10 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %19 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%0 : memref<1x16x16x16xf16, @DDR>) outputs(%2 : memref<1x16x16x16xf16, [@CMX_NN, 0]>) previousDMA(%18 : !VPURegMapped.Index<0:0:1>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%11 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %20 = VPUMI40XX.NNDMA {is_out_of_order, port = 1 : i64} inputs(%cst : memref<1x1x1x4864xui8>) outputs(%5 : memref<1x1x1x4864xui8, [@CMX_NN, 0]>) previousDMA(%19 : !VPURegMapped.Index<0:0:2>) updates(%12 : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>
    %21 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x16x14x14xf16, @DDR>) waits(%13 : !VPURegMapped.Index<0:0:3>) updates(%14 : !VPURegMapped.Index<0:0:4>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %22 = VPUMI40XX.MappedInference dmas((%17, %21) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) invariants(%startIndexes#0 : !VPURegMapped.Index<0:0:0>) variants(%startIndexes#1 : !VPURegMapped.Index<0:0:0>) barriers(%10 : !VPURegMapped.Index<0:0:0>) dmaCount([[4, 1]]) invariantCount([2]) variantCount([2]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(5) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x14x14xf16, @DDR>
  }

  // CHECK: [[CST:%.+]] = const.Declare memref<16xui32>
  // CHECK: [[CST0:%.+]] = const.Declare memref<112xui32>

  // CHECK: [[BUF:%.+]] = VPURT.DeclareBuffer <Register> <788594688> -> memref<64xui32, @Register>
  // CHECK: [[CST1:%.+]] = const.Declare memref<64xui32>

  // CHECK: [[REPRG_DMA:%.+]] = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 256 : i64, srcWidth = 256 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 16 : i64, dstStride = 32 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64, wlmPage = -1 : i64} inputs([[CST1]] : memref<64xui32>) outputs([[BUF]] : memref<64xui32, @Register>)

  // CHECK: {workloadManagementBarrierProgrammingMode = #VPURegMapped.workload_management_barrier_programming_mode<INITIAL_BARRIER_DMAS_SCHEDULED>}
  // CHECK: barrierConfigurationTasks([[CST0]]
  // CHECK-SAME: numOfBarrierReprogrammings([[CST]]
  // CHECK-SAME: barrierConfigurationTasksCount(112)
}
