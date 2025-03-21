//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --link-enqueue-targets %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
module @VPU.SW {
  func.func private @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_hswish.cpp", VPU.kernel_entry = "activation_hswish"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @manyActShaveTasksNoLists() {
  %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %4 = VPUMI40XX.DeclareKernelText kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %5 = VPUMI40XX.DeclareKernelEntry kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %6 = VPUMI40XX.DeclareKernelArgs kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %7 = VPUMI40XX.KernelParams inputs(%2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPURegMapped.Index<0:0:0>

  %r0 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>

  %i0 = VPUMI40XX.ActKernelInvocation range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
  %i1 = VPUMI40XX.ActKernelInvocation previousTask(%i0 : !VPURegMapped.Index<0:0:0>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>

  %i10 = VPUMI40XX.ActKernelInvocation range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:0>
  %i11 = VPUMI40XX.ActKernelInvocation previousTask(%i10 : !VPURegMapped.Index<1:0:0>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:1>

  %b = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <4, -1> -> !VPURegMapped.Index<0:0:0>

  %e0 = VPURegMapped.Enqueue at(%b : !VPURegMapped.Index<0:0:0>) (%i0 -> %i1: <0:0:0> -> <0:0:1>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<ActKernelInvocation>}
  %e1 = VPURegMapped.Enqueue at(%b : !VPURegMapped.Index<0:0:0>) (%i10 -> %i11 : <1:0:0> -> <1:0:1>) -> !VPURegMapped.Index<0:0:1> {taskType = #VPURegMapped.task_type<ActKernelInvocation>}

  %mi = VPUMI40XX.MappedInference actKernelRanges(%r0 : !VPURegMapped.Index<0:0:0>) actKernelInvocations(%i0, %i10 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<1:0:0>) barriers(%b : !VPURegMapped.Index<0:0:0>) workItemTasks(%e0 : !VPURegMapped.Index<0:0:0>) dmaCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([1, 0, 0, 0, 0, 0]) variantCount([2, 2, 0, 0, 0, 0]) actKernelRangesCount([1, 0, 0, 0, 0, 0]) actKernelInvocationsCount([2, 2, 0, 0, 0, 0]) mediaCount(0) barrierCount(1) workItemCount(2) -> !VPURegMapped.Index<0:0:0>
  return
}

//CHECK: VPUMI40XX.ActKernelRange
//CHECK-NOT: taskLinkAttrName

//CHECK: %[[INVO0:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[INVO0_IDX:.+]]

//CHECK: %[[INVO1:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[INVO1_IDX:.+]]

//CHECK: %[[INVO2:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[INVO2_IDX:.+]]

//CHECK: %[[INVO3:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[INVO3_IDX:.+]]

//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: (%[[INVO0]] -> %[[INVO1]] : [[INVO0_IDX]] -> [[INVO1_IDX]])

//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: (%[[INVO2]] -> %[[INVO3]] : [[INVO2_IDX]] -> [[INVO3_IDX]])

//CHECK-NOT: VPURegMapped.Enqueue

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
module @VPU.SW {
  func.func private @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_hswish.cpp", VPU.kernel_entry = "activation_hswish"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @manyActShaveTasks1List() {
  %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %4 = VPUMI40XX.DeclareKernelText kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %5 = VPUMI40XX.DeclareKernelEntry kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %6 = VPUMI40XX.DeclareKernelArgs kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %7 = VPUMI40XX.KernelParams inputs(%2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPURegMapped.Index<0:0:0>

  %r0 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>

  %i0 = VPUMI40XX.ActKernelInvocation range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
  %i1 = VPUMI40XX.ActKernelInvocation previousTask(%i0 : !VPURegMapped.Index<0:0:0>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
  %i2 = VPUMI40XX.ActKernelInvocation previousTask(%i1 : !VPURegMapped.Index<0:0:1>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>

  %i10 = VPUMI40XX.ActKernelInvocation range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:0>
  %i11 = VPUMI40XX.ActKernelInvocation previousTask(%i10 : !VPURegMapped.Index<1:0:0>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:1>

  %b = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <4, -1> -> !VPURegMapped.Index<0:0:0>

  %e0 = VPURegMapped.Enqueue at(%b : !VPURegMapped.Index<0:0:0>) (%i0 -> %i2: <0:0:0> -> <0:0:2>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<ActKernelInvocation>}
  %e1 = VPURegMapped.Enqueue at(%b : !VPURegMapped.Index<0:0:0>) (%i10 -> %i11 : <1:0:0> -> <1:0:1>) -> !VPURegMapped.Index<0:0:1> {taskType = #VPURegMapped.task_type<ActKernelInvocation>}

  %mi = VPUMI40XX.MappedInference actKernelRanges(%r0 : !VPURegMapped.Index<0:0:0>) actKernelInvocations(%i0, %i10 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<1:0:0>) barriers(%b : !VPURegMapped.Index<0:0:0>) workItemTasks(%e0 : !VPURegMapped.Index<0:0:0>) dmaCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([1, 0, 0, 0, 0, 0]) variantCount([2, 2, 0, 0, 0, 0]) actKernelRangesCount([1, 0, 0, 0, 0, 0]) actKernelInvocationsCount([3, 2, 0, 0, 0, 0]) mediaCount(0) barrierCount(1) workItemCount(2) -> !VPURegMapped.Index<0:0:0>
  return
}

//CHECK: VPUMI40XX.ActKernelRange
//CHECK-NOT: taskLinkAttrName

//CHECK: %[[INVO0:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[INVO0_IDX:.+]]

//CHECK: %[[INVO1:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[INVO1_IDX:.+]]

//CHECK: %[[INVO2:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: taskLinkAttrName = #VPURegMapped.IndexType<[[INVO0_IDX]]>
//CHECK-SAME: -> !VPURegMapped.Index[[INVO2_IDX:.+]]

//CHECK: %[[INVO3:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[INVO3_IDX:.+]]

//CHECK: %[[INVO4:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[INVO4_IDX:.+]]

//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: (%[[INVO0]] -> %[[INVO1]] : [[INVO0_IDX]] -> [[INVO1_IDX]])

//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: (%[[INVO3]] -> %[[INVO4]] : [[INVO3_IDX]] -> [[INVO4_IDX]])

//CHECK-NOT: VPURegMapped.Enqueue

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
module @VPU.SW {
  func.func private @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_hswish.cpp", VPU.kernel_entry = "activation_hswish"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @manyActShaveTasks1List() {
  %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %4 = VPUMI40XX.DeclareKernelText kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %5 = VPUMI40XX.DeclareKernelEntry kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %6 = VPUMI40XX.DeclareKernelArgs kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %7 = VPUMI40XX.KernelParams inputs(%2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPURegMapped.Index<0:0:0>

  %r0 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>

  %i0 = VPUMI40XX.ActKernelInvocation range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
  %i1 = VPUMI40XX.ActKernelInvocation previousTask(%i0 : !VPURegMapped.Index<0:0:0>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
  %i2 = VPUMI40XX.ActKernelInvocation previousTask(%i1 : !VPURegMapped.Index<0:0:1>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>
  %i3 = VPUMI40XX.ActKernelInvocation previousTask(%i2 : !VPURegMapped.Index<0:0:2>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:3>
  %i4 = VPUMI40XX.ActKernelInvocation previousTask(%i3 : !VPURegMapped.Index<0:0:3>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:4>

  %i5 = VPUMI40XX.ActKernelInvocation previousTask(%i4 : !VPURegMapped.Index<0:0:4>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:5>
  %i6 = VPUMI40XX.ActKernelInvocation previousTask(%i5 : !VPURegMapped.Index<0:0:5>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:6>
  %i7 = VPUMI40XX.ActKernelInvocation previousTask(%i6 : !VPURegMapped.Index<0:0:6>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:7>
  %i8 = VPUMI40XX.ActKernelInvocation previousTask(%i7 : !VPURegMapped.Index<0:0:7>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:8>

  %i10 = VPUMI40XX.ActKernelInvocation range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:0>
  %i11 = VPUMI40XX.ActKernelInvocation previousTask(%i10 : !VPURegMapped.Index<1:0:0>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:1>
  %i12 = VPUMI40XX.ActKernelInvocation previousTask(%i11 : !VPURegMapped.Index<1:0:1>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:2>
  %i13 = VPUMI40XX.ActKernelInvocation previousTask(%i12 : !VPURegMapped.Index<1:0:2>) range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:3>

  %b = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <4, -1> -> !VPURegMapped.Index<0:0:0>

  %e0 = VPURegMapped.Enqueue at(%b : !VPURegMapped.Index<0:0:0>) (%i0 -> %i5: <0:0:0> -> <0:0:5>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<ActKernelInvocation>}
  %e1 = VPURegMapped.Enqueue at(%b : !VPURegMapped.Index<0:0:0>) (%i6 -> %i8: <0:0:6> -> <0:0:8>) -> !VPURegMapped.Index<0:0:1> {taskType = #VPURegMapped.task_type<ActKernelInvocation>}
  %e2 = VPURegMapped.Enqueue at(%b : !VPURegMapped.Index<0:0:0>) (%i10 -> %i13 : <1:0:0> -> <1:0:3>) -> !VPURegMapped.Index<0:0:2> {taskType = #VPURegMapped.task_type<ActKernelInvocation>}

  %mi = VPUMI40XX.MappedInference actKernelRanges(%r0 : !VPURegMapped.Index<0:0:0>) actKernelInvocations(%i0, %i10 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<1:0:0>) barriers(%b : !VPURegMapped.Index<0:0:0>) workItemTasks(%e0 : !VPURegMapped.Index<0:0:0>) dmaCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([1, 0, 0, 0, 0, 0]) variantCount([2, 2, 0, 0, 0, 0]) actKernelRangesCount([1, 0, 0, 0, 0, 0]) actKernelInvocationsCount([9, 4, 0, 0, 0, 0]) mediaCount(0) barrierCount(1) workItemCount(2) -> !VPURegMapped.Index<0:0:0>
  return
}

//CHECK: VPUMI40XX.ActKernelRange
//CHECK-NOT: taskLinkAttrName

//CHECK: %[[INVO0:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[INVO0_IDX:.+]]

//CHECK: %[[INVO1:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[INVO1_IDX:.+]]

//CHECK: %[[INVO2:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: taskLinkAttrName = #VPURegMapped.IndexType<[[INVO0_IDX]]>
//CHECK-SAME: -> !VPURegMapped.Index[[INVO2_IDX:.+]]

//CHECK: %[[INVO3:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: taskLinkAttrName = #VPURegMapped.IndexType<[[INVO1_IDX]]>
//CHECK-SAME: -> !VPURegMapped.Index[[INVO3_IDX:.+]]

//CHECK: %[[INVO4:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: taskLinkAttrName = #VPURegMapped.IndexType<[[INVO2_IDX]]>
//CHECK-SAME: -> !VPURegMapped.Index[[INVO4_IDX:.+]]

//CHECK: %[[INVO5:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: taskLinkAttrName = #VPURegMapped.IndexType<[[INVO3_IDX]]>
//CHECK-SAME: -> !VPURegMapped.Index[[INVO5_IDX:.+]]

//CHECK: %[[INVO6:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[INVO6_IDX:.+]]

//CHECK: %[[INVO7:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[INVO7_IDX:.+]]

//CHECK: %[[INVO8:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: taskLinkAttrName = #VPURegMapped.IndexType<[[INVO6_IDX]]>
//CHECK-SAME: -> !VPURegMapped.Index[[INVO8_IDX:.+]]

//CHECK: %[[INVO9:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[INVO9_IDX:.+]]

//CHECK: %[[INVO10:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-NOT: taskLinkAttrName
//CHECK-SAME: -> !VPURegMapped.Index[[INVO10_IDX:.+]]

//CHECK: %[[INVO11:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: taskLinkAttrName = #VPURegMapped.IndexType<[[INVO9_IDX]]>
//CHECK-SAME: -> !VPURegMapped.Index[[INVO11_IDX:.+]]

//CHECK: %[[INVO12:.+]] = VPUMI40XX.ActKernelInvocation
//CHECK-SAME: taskLinkAttrName = #VPURegMapped.IndexType<[[INVO10_IDX]]>
//CHECK-SAME: -> !VPURegMapped.Index[[INVO12_IDX:.+]]

//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: (%[[INVO0]] -> %[[INVO1]] : [[INVO0_IDX]] -> [[INVO1_IDX]])

//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: (%[[INVO6]] -> %[[INVO7]] : [[INVO6_IDX]] -> [[INVO7_IDX]])

//CHECK: VPURegMapped.Enqueue
//CHECK-SAME: (%[[INVO9]] -> %[[INVO10]] : [[INVO9_IDX]] -> [[INVO10_IDX]])

//CHECK-NOT: VPURegMapped.Enqueue
