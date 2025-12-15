//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --link-enqueue-targets %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @multiDMA() {
  %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, #NHWC, @DDR>
  %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, #NHWC, @DDR>
  %2 = VPURT.DeclareBuffer <NetworkOutput> [1] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, #NHWC, @DDR>

  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
  %4 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>

  %7 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%3 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
  %8 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%3 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%7 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
  %81 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%3 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%8 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
  %9 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%3 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x16x16x16xf16, #NHWC, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>

  %10 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%4 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:0:0>
  %11 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%4 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) previousDMA(%10 : !VPURegMapped.Index<1:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:0:1>
  %12 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%4 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) outputs(%2 : memref<1x16x16x16xf16, #NHWC, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:1:0>

  %b = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <4, -1> -> !VPURegMapped.Index<0:0:0>

  %e0 = VPURegMapped.Enqueue at(%b : !VPURegMapped.Index<0:0:0>) (%7 -> %8 : <0:0:0> -> <0:0:1>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<DMA>}
  %e1 = VPURegMapped.Enqueue at(%b : !VPURegMapped.Index<0:0:0>) (%81 -> %81 : <0:0:2> -> <0:0:2>) -> !VPURegMapped.Index<0:0:1> {taskType = #VPURegMapped.task_type<DMA>}
  %e2 = VPURegMapped.Enqueue at(%b : !VPURegMapped.Index<0:0:0>) (%9 -> %9 : <0:1:0> -> <0:1:0>) -> !VPURegMapped.Index<0:0:2> {taskType = #VPURegMapped.task_type<DMA>}
  %e3 = VPURegMapped.Enqueue at(%b : !VPURegMapped.Index<0:0:0>) (%10 -> %11 : <1:0:0> -> <1:0:1>) -> !VPURegMapped.Index<0:0:3> {taskType = #VPURegMapped.task_type<DMA>}
  %e4 = VPURegMapped.Enqueue at(%b : !VPURegMapped.Index<0:0:0>) (%12 -> %12 : <1:1:0> -> <1:1:0>) -> !VPURegMapped.Index<0:0:4> {taskType = #VPURegMapped.task_type<DMA>}

  %13 = VPUMI40XX.MappedInference dmas((%7, %9), (%10, %12) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>), (!VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<1:1:0>)) barriers(%b : !VPURegMapped.Index<0:0:0>) workItemTasks(%e0 : !VPURegMapped.Index<0:0:0>) dmaCount([[3, 1], [2, 1], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) actKernelInvocationsCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) mediaCount(0) barrierCount(1) workItemCount(5) -> !VPURegMapped.Index<0:0:0>

  return
}

//CHECK: %[[DMA0:.+]] = VPUMI40XX.NNDMA
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[DMA0_IDX:.+]]

//CHECK: %[[DMA1:.+]] = VPUMI40XX.NNDMA
//CHECK-SAME: taskLinkAttrName = #VPURegMapped.IndexType<[[DMA0_IDX]]>
//CHECK-SAME: -> !VPURegMapped.Index[[DMA1_IDX:.+]]

//CHECK: %[[DMA2:.+]] = VPUMI40XX.NNDMA
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[DMA2_IDX:.+]]

//CHECK: %[[DMA3:.+]] = VPUMI40XX.NNDMA
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[DMA3_IDX:.+]]

//CHECK: %[[DMA4:.+]] = VPUMI40XX.NNDMA
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[DMA4_IDX:.+]]

//CHECK: %[[DMA5:.+]] = VPUMI40XX.NNDMA
//CHECK-SAME: taskLinkAttrName = #VPURegMapped.IndexType<[[DMA4_IDX]]>
//CHECK-SAME: -> !VPURegMapped.Index[[DMA5_IDX:.+]]

//CHECK: %[[DMA6:.+]] = VPUMI40XX.NNDMA
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[DMA6_IDX:.+]]

//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: (%[[DMA0]] -> %[[DMA0]] : [[DMA0_IDX]] -> [[DMA0_IDX]])

//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: (%[[DMA2]] -> %[[DMA2]] : [[DMA2_IDX]] -> [[DMA2_IDX]])

//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: (%[[DMA3]] -> %[[DMA3]] : [[DMA3_IDX]] -> [[DMA3_IDX]])

//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: (%[[DMA4]] -> %[[DMA4]] : [[DMA4_IDX]] -> [[DMA4_IDX]])

//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: (%[[DMA6]] -> %[[DMA6]] : [[DMA6_IDX]] -> [[DMA6_IDX]])

//CHECK-NOT: VPURegMapped.Enqueue
