//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUIP-to-VPUMI37XX %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {
  net.NetworkInfo entryPoint : @race_condition_dma_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x16x16xf16>
    DataInfo "output_1" : tensor<1x16x16x16xf16>
  }
  // CHECK-LABEL: func.func nested @race_condition_dma_f16_f16
  // CHECK-SAME:    [[ARG_0:%[^:]+]]: memref<1x16x16x16xf16, {order = #NHWC}, @DDR>
  // CHECK-SAME:    [[ARG_1:%[^:]+]]: memref<1x16x16x16xf16, {order = #NHWC}, @DDR>
  // CHECK-SAME:    [[ARG_2:%[^:]+]]: memref<1x16x16x16xf16, {order = #NHWC}, @DDR>
  // CHECK-DAG: [[CMX_BUF_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
  // CHECK-DAG: [[CMX_BUF_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>
  func.func nested @race_condition_dma_f16_f16(%arg0: memref<1x16x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x16x16x16xf16, {order = #NHWC}, @DDR>, %arg2: memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) -> (memref<1x16x16x16xf16, {order = #NHWC}, @DDR>, memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>

    %2 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    // CHECK: [[BAR0:%.+]] = VPUMI37XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>

    VPURT.Task updates(%2 : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%arg0 : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) outputs(%0 : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%2 : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%arg0 : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) outputs(%1 : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>) -> memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>
    }

    %3 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    // CHECK: [[BAR1:%.+]] = VPUMI37XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>

    VPURT.Task waits(%2 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%arg0 : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) outputs(%0 : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%2 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%arg0 : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) outputs(%1 : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>) -> memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>
    }

    VPURT.Task waits(%3 : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%0 : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) -> memref<1x16x16x16xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%3 : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%1 : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>) outputs(%arg2 : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) -> memref<1x16x16x16xf16, {order = #NHWC}, @DDR>
    }

    // CHECK: [[NNDMA1_2:%.+]] = VPUMI37XX.NNDMA <{port = 1 : i64}> inputs([[CMX_BUF_1]] : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>) outputs([[ARG_2]] : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) waits([[BAR1]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:2>
    // CHECK: [[NNDMA0_2:%.+]] = VPUMI37XX.NNDMA <{port = 0 : i64}> inputs([[CMX_BUF_0]] : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[ARG_1]] : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) waits([[BAR1]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    // CHECK: [[NNDMA1_1:%.+]] = VPUMI37XX.NNDMA <{port = 1 : i64}> inputs([[ARG_0]] : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) outputs([[CMX_BUF_1]] : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>) nextDMAIdx([[NNDMA1_2]] : !VPURegMapped.Index<0:1:2>) waits([[BAR0]] : !VPURegMapped.Index<0:0:0>) updates([[BAR1]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:1>
    // CHECK: [[NNDMA0_1:%.+]] = VPUMI37XX.NNDMA <{port = 0 : i64}> inputs([[ARG_0]] : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) outputs([[CMX_BUF_0]] : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) nextDMAIdx([[NNDMA0_2]] : !VPURegMapped.Index<0:0:2>) waits([[BAR0]] : !VPURegMapped.Index<0:0:0>) updates([[BAR1]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    // CHECK: [[NNDMA1_0:%.+]] = VPUMI37XX.NNDMA <{port = 1 : i64}> inputs([[ARG_0]] : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) outputs([[CMX_BUF_1]] : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>) nextDMAIdx([[NNDMA1_1]] : !VPURegMapped.Index<0:1:1>) updates([[BAR0]] : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    // CHECK: [[NNDMA0_0:%.+]] = VPUMI37XX.NNDMA <{port = 0 : i64}> inputs([[ARG_0]] : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>) outputs([[CMX_BUF_0]] : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) nextDMAIdx([[NNDMA0_1]] : !VPURegMapped.Index<0:0:1>) updates([[BAR0]] : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    // CHECK: [[MI:%.+]] = VPUMI37XX.MappedInference dmas([[NNDMA0_0]], [[NNDMA1_0]] : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>) barriers([[BAR0]] : !VPURegMapped.Index<0:0:0>) dmaCount([3, 3]) invariantCount(0) variantCount(0) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(2) -> !VPURegMapped.Index<0:0:0>

    return %arg1, %arg2 : memref<1x16x16x16xf16, {order = #NHWC}, @DDR>, memref<1x16x16x16xf16, {order = #NHWC}, @DDR>
  }
}
