//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --resolve-task-location %s | FileCheck %s
// REQUIRES: arch-NPU40XX
VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
module @VPU.SW {
  func.func private @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_hswish.cpp", VPU.kernel_entry = "activation_hswish"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @manyActShaveTasks() {
  %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %4 = VPUMI40XX.DeclareKernelText kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %5 = VPUMI40XX.DeclareKernelEntry kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %6 = VPUMI40XX.DeclareKernelArgs kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %7 = VPUMI40XX.KernelParams inputs(%2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPURegMapped.Index<0:0:0>

  %r0 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
  %r1 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:1>
  %r2 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:2>
  %r3 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:3>
  %r4 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:4>
  %r5 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:5>
  %r6 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:6>
  %r7 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:7>
  %r8 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:8>
  %r9 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:9>
  %r10 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:10>
  %r11 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:11>
  %r12 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:12>
  %r13 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:13>
  %r14 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:14>
  %r15 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:15>
  %r16 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:16>
  %r17 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:17>
  %r18 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:18>
  %r19 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:19>
  %r20 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:20>
  %r21 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:21>
  %r22 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:22>
  %r23 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:23>
  %r24 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:24>
  %r25 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:25>
  %r26 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:26>
  %r27 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:27>
  %r28 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:28>
  %r29 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:29>
  %r30 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:30>
  %r31 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:31>
  %r32 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:32>
  %r33 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:33>
  %r34 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:34>
  %r35 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:35>
  %r36 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:36>
  %r37 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:37>
  %r38 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:38>
  %r39 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:39>
  %r40 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:40>
  %r41 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:41>
  %r42 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:42>
  %r43 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:43>
  %r44 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:44>
  %r45 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:45>
  %r46 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:46>
  %r47 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:47>
  %r48 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:48>
  %r49 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:49>
  %r50 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:50>
  %r51 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:51>
  %r52 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:52>
  %r53 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:53>
  %r54 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:54>
  %r55 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:55>
  %r56 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:56>
  %r57 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:57>
  %r58 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:58>
  %r59 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:59>
  %r60 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:60>
  %r61 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:61>
  %r62 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:62>
  %r63 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:63>
  %r64 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:64>
  %r65 = VPUMI40XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:65>

  %i0 = VPUMI40XX.ActKernelInvocation range_index(%r0 : <0:0:0>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
  %i1 = VPUMI40XX.ActKernelInvocation range_index(%r1 : <0:0:1>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
  %i2 = VPUMI40XX.ActKernelInvocation range_index(%r2 : <0:0:2>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>
  %i3 = VPUMI40XX.ActKernelInvocation range_index(%r3 : <0:0:3>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:3>
  %i4 = VPUMI40XX.ActKernelInvocation range_index(%r4 : <0:0:4>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:4>
  %i5 = VPUMI40XX.ActKernelInvocation range_index(%r5 : <0:0:5>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:5>
  %i6 = VPUMI40XX.ActKernelInvocation range_index(%r6 : <0:0:6>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:6>
  %i7 = VPUMI40XX.ActKernelInvocation range_index(%r7 : <0:0:7>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:7>
  %i8 = VPUMI40XX.ActKernelInvocation range_index(%r8 : <0:0:8>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:8>
  %i9 = VPUMI40XX.ActKernelInvocation range_index(%r9 : <0:0:9>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:9>
  %i10 = VPUMI40XX.ActKernelInvocation range_index(%r10 : <0:0:10>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:10>
  %i11 = VPUMI40XX.ActKernelInvocation range_index(%r11 : <0:0:11>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:11>
  %i12 = VPUMI40XX.ActKernelInvocation range_index(%r12 : <0:0:12>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:12>
  %i13 = VPUMI40XX.ActKernelInvocation range_index(%r13 : <0:0:13>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:13>
  %i14 = VPUMI40XX.ActKernelInvocation range_index(%r14 : <0:0:14>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:14>
  %i15 = VPUMI40XX.ActKernelInvocation range_index(%r15 : <0:0:15>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:15>
  %i16 = VPUMI40XX.ActKernelInvocation range_index(%r16 : <0:0:16>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:16>
  %i17 = VPUMI40XX.ActKernelInvocation range_index(%r17 : <0:0:17>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:17>
  %i18 = VPUMI40XX.ActKernelInvocation range_index(%r18 : <0:0:18>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:18>
  %i19 = VPUMI40XX.ActKernelInvocation range_index(%r19 : <0:0:19>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:19>
  %i20 = VPUMI40XX.ActKernelInvocation range_index(%r20 : <0:0:20>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:20>
  %i21 = VPUMI40XX.ActKernelInvocation range_index(%r21 : <0:0:21>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:21>
  %i22 = VPUMI40XX.ActKernelInvocation range_index(%r22 : <0:0:22>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:22>
  %i23 = VPUMI40XX.ActKernelInvocation range_index(%r23 : <0:0:23>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:23>
  %i24 = VPUMI40XX.ActKernelInvocation range_index(%r24 : <0:0:24>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:24>
  %i25 = VPUMI40XX.ActKernelInvocation range_index(%r25 : <0:0:25>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:25>
  %i26 = VPUMI40XX.ActKernelInvocation range_index(%r26 : <0:0:26>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:26>
  %i27 = VPUMI40XX.ActKernelInvocation range_index(%r27 : <0:0:27>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:27>
  %i28 = VPUMI40XX.ActKernelInvocation range_index(%r28 : <0:0:28>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:28>
  %i29 = VPUMI40XX.ActKernelInvocation range_index(%r29 : <0:0:29>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:29>
  %i30 = VPUMI40XX.ActKernelInvocation range_index(%r30 : <0:0:30>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:30>
  %i31 = VPUMI40XX.ActKernelInvocation range_index(%r31 : <0:0:31>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:31>
  %i32 = VPUMI40XX.ActKernelInvocation range_index(%r32 : <0:0:32>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:32>
  %i33 = VPUMI40XX.ActKernelInvocation range_index(%r33 : <0:0:33>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:33>
  %i34 = VPUMI40XX.ActKernelInvocation range_index(%r34 : <0:0:34>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:34>
  %i35 = VPUMI40XX.ActKernelInvocation range_index(%r35 : <0:0:35>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:35>
  %i36 = VPUMI40XX.ActKernelInvocation range_index(%r36 : <0:0:36>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:36>
  %i37 = VPUMI40XX.ActKernelInvocation range_index(%r37 : <0:0:37>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:37>
  %i38 = VPUMI40XX.ActKernelInvocation range_index(%r38 : <0:0:38>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:38>
  %i39 = VPUMI40XX.ActKernelInvocation range_index(%r39 : <0:0:39>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:39>
  %i40 = VPUMI40XX.ActKernelInvocation range_index(%r40 : <0:0:40>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:40>
  %i41 = VPUMI40XX.ActKernelInvocation range_index(%r41 : <0:0:41>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:41>
  %i42 = VPUMI40XX.ActKernelInvocation range_index(%r42 : <0:0:42>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:42>
  %i43 = VPUMI40XX.ActKernelInvocation range_index(%r43 : <0:0:43>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:43>
  %i44 = VPUMI40XX.ActKernelInvocation range_index(%r44 : <0:0:44>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:44>
  %i45 = VPUMI40XX.ActKernelInvocation range_index(%r45 : <0:0:45>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:45>
  %i46 = VPUMI40XX.ActKernelInvocation range_index(%r46 : <0:0:46>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:46>
  %i47 = VPUMI40XX.ActKernelInvocation range_index(%r47 : <0:0:47>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:47>
  %i48 = VPUMI40XX.ActKernelInvocation range_index(%r48 : <0:0:48>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:48>
  %i49 = VPUMI40XX.ActKernelInvocation range_index(%r49 : <0:0:49>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:49>
  %i50 = VPUMI40XX.ActKernelInvocation range_index(%r50 : <0:0:50>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:50>
  %i51 = VPUMI40XX.ActKernelInvocation range_index(%r51 : <0:0:51>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:51>
  %i52 = VPUMI40XX.ActKernelInvocation range_index(%r52 : <0:0:52>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:52>
  %i53 = VPUMI40XX.ActKernelInvocation range_index(%r53 : <0:0:53>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:53>
  %i54 = VPUMI40XX.ActKernelInvocation range_index(%r54 : <0:0:54>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:54>
  %i55 = VPUMI40XX.ActKernelInvocation range_index(%r55 : <0:0:55>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:55>
  %i56 = VPUMI40XX.ActKernelInvocation range_index(%r56 : <0:0:56>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:56>
  %i57 = VPUMI40XX.ActKernelInvocation range_index(%r57 : <0:0:57>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:57>
  %i58 = VPUMI40XX.ActKernelInvocation range_index(%r58 : <0:0:58>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:58>
  %i59 = VPUMI40XX.ActKernelInvocation range_index(%r59 : <0:0:59>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:59>
  %i60 = VPUMI40XX.ActKernelInvocation range_index(%r60 : <0:0:60>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:60>
  %i61 = VPUMI40XX.ActKernelInvocation range_index(%r61 : <0:0:61>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:61>
  %i62 = VPUMI40XX.ActKernelInvocation range_index(%r62 : <0:0:62>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:62>
  %i63 = VPUMI40XX.ActKernelInvocation range_index(%r63 : <0:0:63>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:63>
  %i64 = VPUMI40XX.ActKernelInvocation range_index(%r64 : <0:0:64>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:64>
  %i65 = VPUMI40XX.ActKernelInvocation range_index(%r65 : <0:0:65>) kernel_params(%7 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:65>

  %mi = VPUMI40XX.MappedInference actKernelRanges(%r0 : !VPURegMapped.Index<0:0:0>) actKernelInvocations(%i0 : !VPURegMapped.Index<0:0:0>) dmaCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([66, 0, 0, 0, 0, 0]) actKernelInvocationsCount([66, 0, 0, 0, 0, 0]) mediaCount(0) barrierCount(0) -> !VPURegMapped.Index<0:0:0>

  return
}

//CHECK: func.func @manyActShaveTasks

//CHECK-DAG: [[TBR0:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:0>
//CHECK-DAG: [[TBR1:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:1>
//CHECK-DAG: [[TBR2:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:2>
//CHECK-DAG: [[TBR3:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:3>
//CHECK-DAG: [[TBR4:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:4>
//CHECK-DAG: [[TBR5:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:5>
//CHECK-DAG: [[TBR6:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:6>
//CHECK-DAG: [[TBR7:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:7>
//CHECK-DAG: [[TBR8:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:8>
//CHECK-DAG: [[TBR9:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:9>
//CHECK-DAG: [[TBR10:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:10>
//CHECK-DAG: [[TBR11:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:11>
//CHECK-DAG: [[TBR12:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:12>
//CHECK-DAG: [[TBR13:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:13>
//CHECK-DAG: [[TBR14:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:14>
//CHECK-DAG: [[TBR15:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:15>
//CHECK-DAG: [[TBR16:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:16>
//CHECK-DAG: [[TBR17:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:17>
//CHECK-DAG: [[TBR18:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:18>
//CHECK-DAG: [[TBR19:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:19>
//CHECK-DAG: [[TBR20:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:20>
//CHECK-DAG: [[TBR21:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:21>
//CHECK-DAG: [[TBR22:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:22>
//CHECK-DAG: [[TBR23:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:23>
//CHECK-DAG: [[TBR24:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:24>
//CHECK-DAG: [[TBR25:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:25>
//CHECK-DAG: [[TBR26:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:26>
//CHECK-DAG: [[TBR27:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:27>
//CHECK-DAG: [[TBR28:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:28>
//CHECK-DAG: [[TBR29:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:29>
//CHECK-DAG: [[TBR30:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:30>
//CHECK-DAG: [[TBR31:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:31>
//CHECK-DAG: [[TBR32:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:32>
//CHECK-DAG: [[TBR33:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:33>
//CHECK-DAG: [[TBR34:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:34>
//CHECK-DAG: [[TBR35:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:35>
//CHECK-DAG: [[TBR36:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:36>
//CHECK-DAG: [[TBR37:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:37>
//CHECK-DAG: [[TBR38:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:38>
//CHECK-DAG: [[TBR39:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:39>
//CHECK-DAG: [[TBR40:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:40>
//CHECK-DAG: [[TBR41:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:41>
//CHECK-DAG: [[TBR42:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:42>
//CHECK-DAG: [[TBR43:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:43>
//CHECK-DAG: [[TBR44:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:44>
//CHECK-DAG: [[TBR45:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:45>
//CHECK-DAG: [[TBR46:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:46>
//CHECK-DAG: [[TBR47:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:47>
//CHECK-DAG: [[TBR48:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:48>
//CHECK-DAG: [[TBR49:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:49>
//CHECK-DAG: [[TBR50:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:50>
//CHECK-DAG: [[TBR51:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:51>
//CHECK-DAG: [[TBR52:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:52>
//CHECK-DAG: [[TBR53:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:53>
//CHECK-DAG: [[TBR54:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:54>
//CHECK-DAG: [[TBR55:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:55>
//CHECK-DAG: [[TBR56:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:56>
//CHECK-DAG: [[TBR57:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:57>
//CHECK-DAG: [[TBR58:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:58>
//CHECK-DAG: [[TBR59:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:59>
//CHECK-DAG: [[TBR60:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:60>
//CHECK-DAG: [[TBR61:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:61>
//CHECK-DAG: [[TBR62:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:62>
//CHECK-DAG: [[TBR63:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelRange> -> !VPURegMapped.Index<0:0:63>

//CHECK-DAG: [[TBI0:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:0>
//CHECK-DAG: [[TBI1:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:1>
//CHECK-DAG: [[TBI2:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:2>
//CHECK-DAG: [[TBI3:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:3>
//CHECK-DAG: [[TBI4:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:4>
//CHECK-DAG: [[TBI5:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:5>
//CHECK-DAG: [[TBI6:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:6>
//CHECK-DAG: [[TBI7:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:7>
//CHECK-DAG: [[TBI8:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:8>
//CHECK-DAG: [[TBI9:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:9>
//CHECK-DAG: [[TBI10:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:10>
//CHECK-DAG: [[TBI11:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:11>
//CHECK-DAG: [[TBI12:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:12>
//CHECK-DAG: [[TBI13:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:13>
//CHECK-DAG: [[TBI14:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:14>
//CHECK-DAG: [[TBI15:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:15>
//CHECK-DAG: [[TBI16:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:16>
//CHECK-DAG: [[TBI17:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:17>
//CHECK-DAG: [[TBI18:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:18>
//CHECK-DAG: [[TBI19:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:19>
//CHECK-DAG: [[TBI20:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:20>
//CHECK-DAG: [[TBI21:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:21>
//CHECK-DAG: [[TBI22:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:22>
//CHECK-DAG: [[TBI23:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:23>
//CHECK-DAG: [[TBI24:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:24>
//CHECK-DAG: [[TBI25:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:25>
//CHECK-DAG: [[TBI26:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:26>
//CHECK-DAG: [[TBI27:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:27>
//CHECK-DAG: [[TBI28:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:28>
//CHECK-DAG: [[TBI29:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:29>
//CHECK-DAG: [[TBI30:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:30>
//CHECK-DAG: [[TBI31:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:31>
//CHECK-DAG: [[TBI32:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:32>
//CHECK-DAG: [[TBI33:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:33>
//CHECK-DAG: [[TBI34:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:34>
//CHECK-DAG: [[TBI35:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:35>
//CHECK-DAG: [[TBI36:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:36>
//CHECK-DAG: [[TBI37:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:37>
//CHECK-DAG: [[TBI38:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:38>
//CHECK-DAG: [[TBI39:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:39>
//CHECK-DAG: [[TBI40:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:40>
//CHECK-DAG: [[TBI41:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:41>
//CHECK-DAG: [[TBI42:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:42>
//CHECK-DAG: [[TBI43:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:43>
//CHECK-DAG: [[TBI44:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:44>
//CHECK-DAG: [[TBI45:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:45>
//CHECK-DAG: [[TBI46:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:46>
//CHECK-DAG: [[TBI47:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:47>
//CHECK-DAG: [[TBI48:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:48>
//CHECK-DAG: [[TBI49:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:49>
//CHECK-DAG: [[TBI50:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:50>
//CHECK-DAG: [[TBI51:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:51>
//CHECK-DAG: [[TBI52:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:52>
//CHECK-DAG: [[TBI53:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:53>
//CHECK-DAG: [[TBI54:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:54>
//CHECK-DAG: [[TBI55:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:55>
//CHECK-DAG: [[TBI56:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:56>
//CHECK-DAG: [[TBI57:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:57>
//CHECK-DAG: [[TBI58:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:58>
//CHECK-DAG: [[TBI59:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:59>
//CHECK-DAG: [[TBI60:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:60>
//CHECK-DAG: [[TBI61:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:61>
//CHECK-DAG: [[TBI62:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:62>
//CHECK-DAG: [[TBI63:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <ActKernelInvocation> -> !VPURegMapped.Index<0:0:63>

//CHECK-DAG: VPURegMapped.TaskBufferLayout

//CHECK: [[R0:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR0]]
//CHECK: [[R1:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR1]]
//CHECK: [[R2:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR2]]
//CHECK: [[R3:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR3]]
//CHECK: [[R4:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR4]]
//CHECK: [[R5:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR5]]
//CHECK: [[R6:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR6]]
//CHECK: [[R7:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR7]]
//CHECK: [[R8:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR8]]
//CHECK: [[R9:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR9]]
//CHECK: [[R10:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR10]]
//CHECK: [[R11:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR11]]
//CHECK: [[R12:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR12]]
//CHECK: [[R13:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR13]]
//CHECK: [[R14:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR14]]
//CHECK: [[R15:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR15]]
//CHECK: [[R16:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR16]]
//CHECK: [[R17:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR17]]
//CHECK: [[R18:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR18]]
//CHECK: [[R19:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR19]]
//CHECK: [[R20:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR20]]
//CHECK: [[R21:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR21]]
//CHECK: [[R22:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR22]]
//CHECK: [[R23:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR23]]
//CHECK: [[R24:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR24]]
//CHECK: [[R25:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR25]]
//CHECK: [[R26:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR26]]
//CHECK: [[R27:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR27]]
//CHECK: [[R28:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR28]]
//CHECK: [[R29:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR29]]
//CHECK: [[R30:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR30]]
//CHECK: [[R31:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR31]]
//CHECK: [[R32:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR32]]
//CHECK: [[R33:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR33]]
//CHECK: [[R34:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR34]]
//CHECK: [[R35:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR35]]
//CHECK: [[R36:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR36]]
//CHECK: [[R37:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR37]]
//CHECK: [[R38:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR38]]
//CHECK: [[R39:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR39]]
//CHECK: [[R40:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR40]]
//CHECK: [[R41:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR41]]
//CHECK: [[R42:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR42]]
//CHECK: [[R43:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR43]]
//CHECK: [[R44:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR44]]
//CHECK: [[R45:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR45]]
//CHECK: [[R46:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR46]]
//CHECK: [[R47:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR47]]
//CHECK: [[R48:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR48]]
//CHECK: [[R49:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR49]]
//CHECK: [[R50:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR50]]
//CHECK: [[R51:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR51]]
//CHECK: [[R52:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR52]]
//CHECK: [[R53:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR53]]
//CHECK: [[R54:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR54]]
//CHECK: [[R55:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR55]]
//CHECK: [[R56:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR56]]
//CHECK: [[R57:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR57]]
//CHECK: [[R58:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR58]]
//CHECK: [[R59:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR59]]
//CHECK: [[R60:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR60]]
//CHECK: [[R61:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR61]]
//CHECK: [[R62:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR62]]
//CHECK: [[R63:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR63]]
//CHECK: [[R64:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR0]]
//CHECK: [[R65:%.*]] = VPUMI40XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TBR1]]

//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI0]] : !VPURegMapped.Index<0:0:0>)
    //CHECK-SAME: range_index([[R0]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI1]] : !VPURegMapped.Index<0:0:1>)
    //CHECK-SAME: range_index([[R1]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI2]] : !VPURegMapped.Index<0:0:2>)
    //CHECK-SAME: range_index([[R2]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI3]] : !VPURegMapped.Index<0:0:3>)
    //CHECK-SAME: range_index([[R3]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI4]] : !VPURegMapped.Index<0:0:4>)
    //CHECK-SAME: range_index([[R4]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI5]] : !VPURegMapped.Index<0:0:5>)
    //CHECK-SAME: range_index([[R5]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI6]] : !VPURegMapped.Index<0:0:6>)
    //CHECK-SAME: range_index([[R6]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI7]] : !VPURegMapped.Index<0:0:7>)
    //CHECK-SAME: range_index([[R7]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI8]] : !VPURegMapped.Index<0:0:8>)
    //CHECK-SAME: range_index([[R8]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI9]] : !VPURegMapped.Index<0:0:9>)
    //CHECK-SAME: range_index([[R9]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI10]] : !VPURegMapped.Index<0:0:10>)
    //CHECK-SAME: range_index([[R10]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI11]] : !VPURegMapped.Index<0:0:11>)
    //CHECK-SAME: range_index([[R11]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI12]] : !VPURegMapped.Index<0:0:12>)
    //CHECK-SAME: range_index([[R12]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI13]] : !VPURegMapped.Index<0:0:13>)
    //CHECK-SAME: range_index([[R13]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI14]] : !VPURegMapped.Index<0:0:14>)
    //CHECK-SAME: range_index([[R14]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI15]] : !VPURegMapped.Index<0:0:15>)
    //CHECK-SAME: range_index([[R15]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI16]] : !VPURegMapped.Index<0:0:16>)
    //CHECK-SAME: range_index([[R16]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI17]] : !VPURegMapped.Index<0:0:17>)
    //CHECK-SAME: range_index([[R17]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI18]] : !VPURegMapped.Index<0:0:18>)
    //CHECK-SAME: range_index([[R18]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI19]] : !VPURegMapped.Index<0:0:19>)
    //CHECK-SAME: range_index([[R19]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI20]] : !VPURegMapped.Index<0:0:20>)
    //CHECK-SAME: range_index([[R20]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI21]] : !VPURegMapped.Index<0:0:21>)
    //CHECK-SAME: range_index([[R21]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI22]] : !VPURegMapped.Index<0:0:22>)
    //CHECK-SAME: range_index([[R22]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI23]] : !VPURegMapped.Index<0:0:23>)
    //CHECK-SAME: range_index([[R23]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI24]] : !VPURegMapped.Index<0:0:24>)
    //CHECK-SAME: range_index([[R24]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI25]] : !VPURegMapped.Index<0:0:25>)
    //CHECK-SAME: range_index([[R25]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI26]] : !VPURegMapped.Index<0:0:26>)
    //CHECK-SAME: range_index([[R26]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI27]] : !VPURegMapped.Index<0:0:27>)
    //CHECK-SAME: range_index([[R27]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI28]] : !VPURegMapped.Index<0:0:28>)
    //CHECK-SAME: range_index([[R28]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI29]] : !VPURegMapped.Index<0:0:29>)
    //CHECK-SAME: range_index([[R29]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI30]] : !VPURegMapped.Index<0:0:30>)
    //CHECK-SAME: range_index([[R30]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI31]] : !VPURegMapped.Index<0:0:31>)
    //CHECK-SAME: range_index([[R31]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI32]] : !VPURegMapped.Index<0:0:32>)
    //CHECK-SAME: range_index([[R32]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI33]] : !VPURegMapped.Index<0:0:33>)
    //CHECK-SAME: range_index([[R33]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI34]] : !VPURegMapped.Index<0:0:34>)
    //CHECK-SAME: range_index([[R34]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI35]] : !VPURegMapped.Index<0:0:35>)
    //CHECK-SAME: range_index([[R35]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI36]] : !VPURegMapped.Index<0:0:36>)
    //CHECK-SAME: range_index([[R36]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI37]] : !VPURegMapped.Index<0:0:37>)
    //CHECK-SAME: range_index([[R37]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI38]] : !VPURegMapped.Index<0:0:38>)
    //CHECK-SAME: range_index([[R38]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI39]] : !VPURegMapped.Index<0:0:39>)
    //CHECK-SAME: range_index([[R39]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI40]] : !VPURegMapped.Index<0:0:40>)
    //CHECK-SAME: range_index([[R40]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI41]] : !VPURegMapped.Index<0:0:41>)
    //CHECK-SAME: range_index([[R41]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI42]] : !VPURegMapped.Index<0:0:42>)
    //CHECK-SAME: range_index([[R42]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI43]] : !VPURegMapped.Index<0:0:43>)
    //CHECK-SAME: range_index([[R43]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI44]] : !VPURegMapped.Index<0:0:44>)
    //CHECK-SAME: range_index([[R44]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI45]] : !VPURegMapped.Index<0:0:45>)
    //CHECK-SAME: range_index([[R45]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI46]] : !VPURegMapped.Index<0:0:46>)
    //CHECK-SAME: range_index([[R46]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI47]] : !VPURegMapped.Index<0:0:47>)
    //CHECK-SAME: range_index([[R47]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI48]] : !VPURegMapped.Index<0:0:48>)
    //CHECK-SAME: range_index([[R48]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI49]] : !VPURegMapped.Index<0:0:49>)
    //CHECK-SAME: range_index([[R49]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI50]] : !VPURegMapped.Index<0:0:50>)
    //CHECK-SAME: range_index([[R50]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI51]] : !VPURegMapped.Index<0:0:51>)
    //CHECK-SAME: range_index([[R51]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI52]] : !VPURegMapped.Index<0:0:52>)
    //CHECK-SAME: range_index([[R52]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI53]] : !VPURegMapped.Index<0:0:53>)
    //CHECK-SAME: range_index([[R53]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI54]] : !VPURegMapped.Index<0:0:54>)
    //CHECK-SAME: range_index([[R54]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI55]] : !VPURegMapped.Index<0:0:55>)
    //CHECK-SAME: range_index([[R55]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI56]] : !VPURegMapped.Index<0:0:56>)
    //CHECK-SAME: range_index([[R56]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI57]] : !VPURegMapped.Index<0:0:57>)
    //CHECK-SAME: range_index([[R57]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI58]] : !VPURegMapped.Index<0:0:58>)
    //CHECK-SAME: range_index([[R58]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI59]] : !VPURegMapped.Index<0:0:59>)
    //CHECK-SAME: range_index([[R59]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI60]] : !VPURegMapped.Index<0:0:60>)
    //CHECK-SAME: range_index([[R60]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI61]] : !VPURegMapped.Index<0:0:61>)
    //CHECK-SAME: range_index([[R61]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI62]] : !VPURegMapped.Index<0:0:62>)
    //CHECK-SAME: range_index([[R62]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI63]] : !VPURegMapped.Index<0:0:63>)
    //CHECK-SAME: range_index([[R63]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI0]] : !VPURegMapped.Index<0:0:0>)
    //CHECK-SAME: range_index([[R64]]
//CHECK: VPUMI40XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TBI1]] : !VPURegMapped.Index<0:0:1>)
    //CHECK-SAME: range_index([[R65]]
