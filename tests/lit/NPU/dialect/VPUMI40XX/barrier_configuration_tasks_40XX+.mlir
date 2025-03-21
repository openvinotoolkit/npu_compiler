//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --add-initial-barrier-confuguration-ops %s | FileCheck %s
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

  // CHECK: [[CST0:%.+]] = const.Declare memref<16xui32>
  // CHECK: [[CST1:%.+]] = const.Declare memref<112xui32>

  // CHECK: barrierConfigurationTasks([[CST1]]
  // CHECK-SAME: numOfBarrierReprogrammings([[CST0]]
  // CHECK-SAME: barrierConfigurationTasksCount(112)
}
