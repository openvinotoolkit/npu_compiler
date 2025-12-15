//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true enable-sw-kernel-fifo-per-shave-engine=true" --link-enqueue-targets %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
module @VPU.SW {
  func.func private @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_hswish.cpp", VPU.kernel_entry = "activation_hswish"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @multiShave() {
  %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %4 = VPUMI40XX.DeclareKernelText kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %5 = VPUMI40XX.DeclareKernelEntry kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %6 = VPUMI40XX.DeclareKernelArgs kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %7 = VPUMI40XX.KernelParams inputs(%2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params([0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]) -> !VPURegMapped.Index<0:0:0>

  %r0 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>

  %i0 = VPUMI40XX.ActKernelInvocation range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
  %i1 = VPUMI40XX.ActKernelInvocation previousTask(%i0 : !VPURegMapped.Index<0:0:0>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
  %i2 = VPUMI40XX.ActKernelInvocation previousTask(%i1 : !VPURegMapped.Index<0:0:1>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>
  %i3 = VPUMI40XX.ActKernelInvocation previousTask(%i2 : !VPURegMapped.Index<0:0:2>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:3>

  %i4 = VPUMI40XX.ActKernelInvocation range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:1:0>
  %i5 = VPUMI40XX.ActKernelInvocation previousTask(%i4 : !VPURegMapped.Index<0:1:0>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:1:1>

  %i6 = VPUMI40XX.ActKernelInvocation range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:0>

  %i7 = VPUMI40XX.ActKernelInvocation range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:1:0>
  %i8 = VPUMI40XX.ActKernelInvocation previousTask(%i7 : !VPURegMapped.Index<1:1:0>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:1:1>
  %i9 = VPUMI40XX.ActKernelInvocation previousTask(%i8 : !VPURegMapped.Index<1:1:1>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:1:2>

  %b = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <4, -1> -> !VPURegMapped.Index<0:0:0>

  %e0 = VPURegMapped.Enqueue at(%b : !VPURegMapped.Index<0:0:0>) (%i0 -> %i2: <0:0:0> -> <0:0:2>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<ActKernelInvocation>}
  %e1 = VPURegMapped.Enqueue at(%b : !VPURegMapped.Index<0:0:0>) (%i3 -> %i3: <0:0:3> -> <0:0:3>) -> !VPURegMapped.Index<0:0:1> {taskType = #VPURegMapped.task_type<ActKernelInvocation>}
  %e2 = VPURegMapped.Enqueue at(%b : !VPURegMapped.Index<0:0:0>) (%i4 -> %i5 : <0:1:0> -> <0:1:1>) -> !VPURegMapped.Index<0:0:2> {taskType = #VPURegMapped.task_type<ActKernelInvocation>}
  %e3 = VPURegMapped.Enqueue at(%b : !VPURegMapped.Index<0:0:0>) (%i6 -> %i6 : <1:0:0> -> <1:0:0>) -> !VPURegMapped.Index<0:0:3> {taskType = #VPURegMapped.task_type<ActKernelInvocation>}
  %e4 = VPURegMapped.Enqueue at(%b : !VPURegMapped.Index<0:0:0>) (%i7 -> %i7 : <1:1:0> -> <1:1:0>) -> !VPURegMapped.Index<0:0:4> {taskType = #VPURegMapped.task_type<ActKernelInvocation>}
  %e5 = VPURegMapped.Enqueue at(%b : !VPURegMapped.Index<0:0:0>) (%i8 -> %i9 : <1:1:1> -> <1:1:2>) -> !VPURegMapped.Index<0:0:5> {taskType = #VPURegMapped.task_type<ActKernelInvocation>}

  %mi = VPUMI40XX.MappedInference actKernelRanges((%r0) : (!VPURegMapped.Index<0:0:0>)) actKernelInvocations((%i0, %i4), (%i6, %i7) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>), (!VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<1:1:0>)) barriers(%b : !VPURegMapped.Index<0:0:0>) workItemTasks(%e0 : !VPURegMapped.Index<0:0:0>) dmaCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([1, 0, 0, 0, 0, 0]) variantCount([2, 2, 0, 0, 0, 0]) actKernelRangesCount([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) actKernelInvocationsCount([[4, 2], [1, 3], [0, 0], [0, 0], [0, 0], [0, 0]]) mediaCount(0) barrierCount(1) workItemCount(6) -> !VPURegMapped.Index<0:0:0>
  return
}

//CHECK: VPUMI40XX.ActKernelRange
//CHECK-NOT: taskLinkAttrName

//CHECK: %[[INVO0:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[INVO0_IDX:.+]]

//CHECK: %[[INVO1:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: taskLinkAttrName = #VPURegMapped.IndexType<[[INVO0_IDX]]>
//CHECK-SAME: -> !VPURegMapped.Index[[INVO1_IDX:.+]]

//CHECK: %[[INVO2:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: taskLinkAttrName = #VPURegMapped.IndexType<[[INVO1_IDX]]>
//CHECK-SAME: -> !VPURegMapped.Index[[INVO2_IDX:.+]]

//CHECK: %[[INVO3:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[INVO3_IDX:.+]]

//CHECK: %[[INVO4:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[INVO4_IDX:.+]]

//CHECK: %[[INVO5:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: taskLinkAttrName = #VPURegMapped.IndexType<[[INVO4_IDX]]>
//CHECK-SAME: -> !VPURegMapped.Index[[INVO5_IDX:.+]]

//CHECK: %[[INVO6:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[INVO6_IDX:.+]]

//CHECK: %[[INVO7:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[INVO7_IDX:.+]]

//CHECK: %[[INVO8:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[INVO8_IDX:.+]]

//CHECK: %[[INVO9:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: taskLinkAttrName = #VPURegMapped.IndexType<[[INVO8_IDX]]>
//CHECK-SAME: -> !VPURegMapped.Index[[INVO9_IDX:.+]]

//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: (%[[INVO0]] -> %[[INVO0]] : [[INVO0_IDX]] -> [[INVO0_IDX]])

//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: (%[[INVO3]] -> %[[INVO3]] : [[INVO3_IDX]] -> [[INVO3_IDX]])

//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: (%[[INVO4]] -> %[[INVO4]] : [[INVO4_IDX]] -> [[INVO4_IDX]])

//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: (%[[INVO6]] -> %[[INVO6]] : [[INVO6_IDX]] -> [[INVO6_IDX]])

//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: (%[[INVO7]] -> %[[INVO7]] : [[INVO7_IDX]] -> [[INVO7_IDX]])

//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: (%[[INVO8]] -> %[[INVO8]] : [[INVO8_IDX]] -> [[INVO8_IDX]])

//CHECK-NOT: VPURegMapped.Enqueue
