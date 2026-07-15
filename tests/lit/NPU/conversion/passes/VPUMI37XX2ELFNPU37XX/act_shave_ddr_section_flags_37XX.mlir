//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUMI37XX-to-ELF %s | FileCheck %s
// REQUIRES: platform-NPU3720

module @act_shave_weights_access {
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_Minimum(memref<*xf16>, memref<*xf16>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "eltwise_min.cpp", VPU.kernel_entry = "eltwise_min", VPU.task_type = @COMPUTE}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

  net.NetworkInfo entryPoint : @act_shave_weights_access inputsInfo : {
    DataInfo "input_0" : tensor<1x32x32x514xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x32x514xf16>
  }
  func.func nested @act_shave_weights_access() {
    %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x32x32x514xf16, @DDR>
    %cst = const.Declare memref<1x32x32x514xf16> = dense<1.000000e+00> : tensor<1x32x32x514xf16>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x32x514xf16, [@CMX_NN, 0]>
    %2 = VPUMI37XX.DeclareKernelText kernel_path("eltwise_min_3720xx") -> !VPURegMapped.Index<0:0:0>
    %3 = VPUMI37XX.DeclareKernelEntry kernel_path("eltwise_min_3720xx") -> !VPURegMapped.Index<0:0:0>
    %4 = VPUMI37XX.DeclareKernelArgs kernel_path("eltwise_min_3720xx") -> !VPURegMapped.Index<0:0:0>
    %5 = VPUMI37XX.ActKernelRange kernel_text_index(%2 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%3 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
    %7 = VPUMI37XX.KernelParams inputs(%0, %cst : memref<1x32x32x514xf16, @DDR>, memref<1x32x32x514xf16>) outputs(%1 : memref<1x32x32x514xf16, [@CMX_NN, 0]>) kernel_type("eltwise_min") kernel_params(dense<0> : vector<108xui8>) -> !VPURegMapped.Index<0:0:0>
    %6 = VPUMI37XX.ActKernelInvocation range_index(%5 : <0:0:0>) params_index(%7 : !VPURegMapped.Index<0:0:0>) tile(0) start_after(3) clean_after(2) -> !VPURegMapped.Index<0:0:0>
    %8 = VPUMI37XX.MappedInference actKernelRanges(%5 : !VPURegMapped.Index<0:0:0>) actKernelInvocations(%6 : !VPURegMapped.Index<0:0:0>) dmaCount([0, 0]) invariantCount(0) variantCount(0) actKernelRangesCount(1) actKernelInvocationsCount(1) barrierCount(0) -> !VPURegMapped.Index<0:0:0>
    return
  }
}

// CHECK: [[SEC_SCRATCH:%.+]] = ELFNPU37XX.CreateLogicalSection secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_PROC_SHAVE") {{{.+}} secName = ".data.BuffersIO"} -> !ELFNPU37XX.Section

// CHECK: [[SEC_CONST:%.+]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_SHAVE") {{{.+}} secName = ".data.ConstIO"} -> !ELFNPU37XX.Section

// -----

module @act_shave_scratch_access {
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_Minimum(memref<*xf16>, memref<*xf16>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "eltwise_min.cpp", VPU.kernel_entry = "eltwise_min", VPU.task_type = @COMPUTE}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

  net.NetworkInfo entryPoint : @act_shave_scratch_access inputsInfo : {
    DataInfo "input_0" : tensor<1x32x32x514xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x32x514xf16>
  }
  func.func nested @act_shave_scratch_access() {
    %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x32x32x514xf16, @DDR>
    %var = VPURT.DeclareBuffer <DDR> <1052672> -> memref<1x32x32x514xf16, @DDR>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x32x514xf16, [@CMX_NN, 0]>
    %2 = VPUMI37XX.DeclareKernelText kernel_path("eltwise_min_3720xx") -> !VPURegMapped.Index<0:0:0>
    %3 = VPUMI37XX.DeclareKernelEntry kernel_path("eltwise_min_3720xx") -> !VPURegMapped.Index<0:0:0>
    %4 = VPUMI37XX.DeclareKernelArgs kernel_path("eltwise_min_3720xx") -> !VPURegMapped.Index<0:0:0>
    %5 = VPUMI37XX.ActKernelRange kernel_text_index(%2 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%3 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
    %7 = VPUMI37XX.KernelParams inputs(%0, %var : memref<1x32x32x514xf16, @DDR>, memref<1x32x32x514xf16, @DDR>) outputs(%1 : memref<1x32x32x514xf16, [@CMX_NN, 0]>) kernel_type("eltwise_min") kernel_params(dense<0> : vector<108xui8>) -> !VPURegMapped.Index<0:0:0>
    %6 = VPUMI37XX.ActKernelInvocation range_index(%5 : <0:0:0>) params_index(%7 : !VPURegMapped.Index<0:0:0>) tile(0) start_after(3) clean_after(2) -> !VPURegMapped.Index<0:0:0>
    %8 = VPUMI37XX.MappedInference actKernelRanges(%5 : !VPURegMapped.Index<0:0:0>) actKernelInvocations(%6 : !VPURegMapped.Index<0:0:0>) dmaCount([0, 0]) invariantCount(0) variantCount(0) actKernelRangesCount(1) actKernelInvocationsCount(1) barrierCount(0) -> !VPURegMapped.Index<0:0:0>
    return
  }
}

// CHECK: [[SEC_SCRATCH:%.+]] = ELFNPU37XX.CreateLogicalSection secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_PROC_SHAVE") {{{.+}} secName = ".data.BuffersIO"} -> !ELFNPU37XX.Section

// CHECK: [[SEC_CONST:%.+]] = ELFNPU37XX.CreateSection secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {{{.+}} secName = ".data.ConstIO"} -> !ELFNPU37XX.Section
