//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --add-bootstrap-ops %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  IE.ExecutorResource 1 of @DMA_NN
  IE.TileResource 1 of @NCE at 6.000000e+02 MHz
  IE.CNNNetwork entryPoint : @twoDma inputsInfo : {
    DataInfo "input_0" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x16x16xf16>
    DataInfo "output_1" : tensor<1x16x16x16xf16>
  }

  func.func @twoDma() {
    %0 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:1>
    %2 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:2>
    %3 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:3>
    %4 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:1:0>
    %5 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:1:1>

    %6 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, #NHWC, @DDR>
    %7 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, #NHWC, @DDR>
    %8 = VPURT.DeclareBuffer <NetworkOutput> [1] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, #NHWC, @DDR>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %10 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>

    %11 = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    %12 = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>

    %13 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>) inputs(%6 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%9 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) updates(%11 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %14 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%1 : !VPURegMapped.Index<0:0:1>) inputs(%6 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%9 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%13 : !VPURegMapped.Index<0:0:0>) waits(%11 : !VPURegMapped.Index<0:0:0>) updates(%12 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %15 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%2 : !VPURegMapped.Index<0:0:2>) inputs(%6 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%10 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) previousDMA(%14 : !VPURegMapped.Index<0:0:1>) updates(%11 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %16 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%3 : !VPURegMapped.Index<0:0:3>) inputs(%6 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%10 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) previousDMA(%15 : !VPURegMapped.Index<0:0:2>) waits(%11 : !VPURegMapped.Index<0:0:0>) updates(%12 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>

    %17 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%4 : !VPURegMapped.Index<0:1:0>) inputs(%9 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%7 : memref<1x16x16x16xf16, #NHWC, @DDR>) waits(%12 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %18 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%5 : !VPURegMapped.Index<0:1:1>) inputs(%10 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) outputs(%8 : memref<1x16x16x16xf16, #NHWC, @DDR>) previousDMA(%17 : !VPURegMapped.Index<0:1:0>) waits(%12 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:1>

    %19 = VPUMI40XX.MappedInference dmas((%13, %17) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) barriers(%11 : !VPURegMapped.Index<0:0:0>) dmaCount([[4, 2], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([0, 0, 0, 0, 0, 0]) actKernelInvocationsCount([0, 0, 0, 0, 0, 0]) mediaCount(0) barrierCount(2) workItemCount(1) bootstrapTasksCount(0)-> !VPURegMapped.Index<0:0:0>
    return
  }

  // CHECK: [[VAL0:%.*]] = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8} <0, -1> -> !VPURegMapped.Index<0:0:0>
  // CHECK: [[VAL1:%.*]] = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8} <1, -1> -> !VPURegMapped.Index<0:0:1>
  // CHECK: VPUMI40XX.Bootstrap inputs([[VAL0]] : <0:0:0>) -> !VPURegMapped.Index<0:0:0>
  // CHECK: VPUMI40XX.Bootstrap inputs([[VAL1]] : <0:0:1>) -> !VPURegMapped.Index<0:0:1>
  // CHECK: VPURegMapped.Enqueue
  // CHECK-SAME <0:0:0> -> <0:0:3>
  // CHECK: VPURegMapped.Enqueue
  // CHECK-SAME <0:1:0> -> <0:1:1>
  // CHECK: workItemCount(2)
  // CHECK-SAME: bootstrapTasksCount(2)
  // CHECK-SAME: bootsrapWorkItemsCount(2)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module {
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 1 of @DMA_NN
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork entryPoint : @DmaAndVarint inputsInfo : {
    DataInfo "input" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x14x14xf16>
  }
  func.func @DmaAndVarint(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<1x16x14x14xf16, @DDR>) -> memref<1x16x14x14xf16, @DDR> {
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
    %10 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <0, -1> -> !VPURegMapped.Index<0:0:0>
    %11 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 2 : ui8} <1, -1> -> !VPURegMapped.Index<0:0:1>
    %12 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <2, -1> -> !VPURegMapped.Index<0:0:2>
    %13 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, isFinalBarrier, producer_count = 1 : ui8} <3, -1> -> !VPURegMapped.Index<0:0:3>
    %14 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%0 : memref<1x16x16x16xf16, @DDR>) outputs(%2 : memref<1x16x16x16xf16, [@CMX_NN, 0]>) updates(%10 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %15 = VPUMI40XX.NNDMA {is_out_of_order, port = 1 : i64} inputs(%cst : memref<1x1x1x4864xui8>) outputs(%5 : memref<1x1x1x4864xui8, [@CMX_NN, 0]>) previousDMA(%14 : !VPURegMapped.Index<0:0:0>) updates(%11 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %16 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x16x14x14xf16, @DDR>) waits(%12 : !VPURegMapped.Index<0:0:2>) updates(%13 : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %17 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64} input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%11 : !VPURegMapped.Index<0:0:1>) -> <0:0:0> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %18 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} previousTask(%17 : !VPURegMapped.Index<0:0:0>) input(%7 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%12 : !VPURegMapped.Index<0:0:2>) -> <0:0:1> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %19 = VPUMI40XX.DPUVariant calls(%17 : <0:0:0>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
    %20 = VPUMI40XX.DPUVariant previousTask(%19 : !VPURegMapped.Index<0:0:0>) calls(%18 : <0:0:1>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
    %21 = VPUMI40XX.MappedInference dmas((%14, %16) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) invariants(%17 : !VPURegMapped.Index<0:0:0>) variants(%19 : !VPURegMapped.Index<0:0:0>) barriers(%10 : !VPURegMapped.Index<0:0:0>) dmaCount([[2, 1]]) invariantCount([2]) variantCount([2]) actKernelRangesCount([0]) actKernelInvocationsCount([0]) mediaCount(0) barrierCount(4) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x14x14xf16, @DDR>
  }

  // CHECK: VPURegMapped.Enqueue
  // CHECK-SAME <0:0:0> -> <0:0:1>
  // CHECK-SAME taskType = #VPURegMapped.task_type<DMA>
  // CHECK: VPURegMapped.Enqueue
  // CHECK-SAME <0:1:0> -> <0:1:0>
  // CHECK-SAME taskType = #VPURegMapped.task_type<DMA>
  // CHECK: VPURegMapped.Enqueue
  // CHECK-SAME <0:0:0> -> <0:0:1>
  // CHECK-SAME   taskType = #VPURegMapped.task_type<DPUVariant>

  // CHECK: workItemCount(3)
  // CHECK-SAME: bootstrapTasksCount(4)
  // CHECK-SAME: bootsrapWorkItemsCount(3)
}

// -----

module attributes {VPU.arch = #VPU.arch_kind<NPU40XX>} {
  IE.ExecutorResource 1 of @DMA_NN
  IE.TileResource 1 of @NCE at 6.000000e+02 MHz
  IE.CNNNetwork entryPoint : @single_hswish inputsInfo : {
    DataInfo "input" : tensor<1x1000xf16>
  } outputsInfo : {
    DataInfo "hswish" : tensor<1x1000xf16>
  }
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
  module @VPU.SW {
    func.func private @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_hswish.cpp", VPU.kernel_entry = "activation_hswish"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

  func.func @single_hswish() {
    %0 = VPURegMapped.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:0:0>
    %1 = VPURegMapped.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:0>
    %2 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %3 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:1>

    %4 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x1x1x1000xf16>
    %5 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x1x1x1000xf16>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    %8 = VPUMI40XX.DeclareKernelText kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
    %9 = VPUMI40XX.DeclareKernelEntry kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
    %10 = VPUMI40XX.DeclareKernelArgs kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
    %11 = VPUMI40XX.KernelParams inputs(%6 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%7 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPURegMapped.Index<0:0:0>

    %12 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    %13 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>

    %14 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%2 : !VPURegMapped.Index<0:0:0>) inputs(%4 : memref<1x1x1x1000xf16>) outputs(%6 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) updates(%12 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %15 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%3 : !VPURegMapped.Index<0:0:1>) inputs(%7 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%5 : memref<1x1x1x1000xf16>) previousDMA(%14 : !VPURegMapped.Index<0:0:0>) waits(%13 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>

    %16 = VPUMI40XX.ActKernelRange taskLocation(%0 : !VPURegMapped.Index<0:0:0>) kernel_text_index(%8 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%10 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%9 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>

    %17 = VPUMI40XX.ActKernelInvocation taskLocation(%1 : !VPURegMapped.Index<0:0:0>) range_index(%16 : <0:0:0>) kernel_params(%11 : <0:0:0>) waits(%12 : !VPURegMapped.Index<0:0:0>) updates(%13 : !VPURegMapped.Index<0:0:1>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>

    %18 = VPUMI40XX.MappedInference dmas((%14, %15) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:1>)) actKernelRanges(%16 : !VPURegMapped.Index<0:0:0>) actKernelInvocations(%17 : !VPURegMapped.Index<0:0:0>) barriers(%12 : !VPURegMapped.Index<0:0:0>) dmaCount([[1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([1, 0, 0, 0, 0, 0]) actKernelInvocationsCount([1, 0, 0, 0, 0, 0]) mediaCount(0) barrierCount(2) -> !VPURegMapped.Index<0:0:0>
    VPUMI40XX.OpRanges
  }

  // CHECK: VPURegMapped.Enqueue
  // CHECK-SAME <0:0:0> -> <0:0:1>
  // CHECK-SAME taskType = #VPURegMapped.task_type<DMA>
  // CHECK: VPURegMapped.Enqueue
  // CHECK-SAME <0:0:0> -> <0:0:0>
  // CHECK-SAME taskType = #VPURegMapped.task_type<ActKernelInvocation>

  // CHECK: workItemCount(2)
  // CHECK-SAME: bootstrapTasksCount(2)
  // CHECK-SAME: bootsrapWorkItemsCount(2)
}
