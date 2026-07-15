//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform%" %s
// this test can only be (correctly) run manually until E#48620 is solved
// REQUIRES: platform-NPU3720

module @Test {

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1000xf16>
  } outputsInfo : {
    DataInfo "sigmoid" : tensor<1x1000xf16>
  }
  module @VPU.SW {
    func.func nested @builtin_sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_sigmoid.cpp", VPU.kernel_entry = "activation_sigmoid"}
    func.func nested @builtin_softmax(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
  }
  func.func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <4000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <6000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %4 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    %5 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>
    %6 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<2, -1> -> !VPURegMapped.Index<0:0:2>
    %7 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<3, -1> -> !VPURegMapped.Index<0:0:3>
    %27 = VPUMI37XX.NNDMA <{port = 0 : i64}> inputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x1x1x1000xf16>) waits(%7 : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %8 = VPUMI37XX.NNDMA <{port = 0 : i64}> inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) nextDMAIdx(%27 : !VPURegMapped.Index<0:0:1>) updates(%4 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %9 = VPUMI37XX.DeclareKernelText kernel_path("activation_sigmoid") -> !VPURegMapped.Index<0:0:0>
    %10 = VPUMI37XX.DeclareKernelArgs kernel_path("activation_sigmoid") -> !VPURegMapped.Index<0:0:0>
    %11 = VPUMI37XX.DeclareKernelEntry kernel_path("activation_sigmoid") -> !VPURegMapped.Index<0:0:0>
    %12 = VPUMI37XX.ActKernelRange kernel_text_index(%9 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%10 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%11 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
    %13 = VPUMI37XX.ActKernelInvocation range_index(%12 : <0:0:0>) waits(%4 : !VPURegMapped.Index<0:0:0>) updates(%5 : !VPURegMapped.Index<0:0:1>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %14 = VPUMI37XX.KernelParams inputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_sigmoid") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPURegMapped.Index<0:0:0>
    %15 = VPUMI37XX.DeclareKernelText kernel_path("softmax") -> !VPURegMapped.Index<0:0:1>
    %16 = VPUMI37XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:1>
    %17 = VPUMI37XX.DeclareKernelEntry kernel_path("softmax") -> !VPURegMapped.Index<0:0:1>
    %18 = VPUMI37XX.ActKernelRange kernel_text_index(%15 : !VPURegMapped.Index<0:0:1>) kernel_args_index(%16 : !VPURegMapped.Index<0:0:1>) kernel_entry_index(%17 : !VPURegMapped.Index<0:0:1>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:1>
    %19 = VPUMI37XX.ActKernelInvocation range_index(%18 : <0:0:1>) waits(%5 : !VPURegMapped.Index<0:0:1>) updates(%6 : !VPURegMapped.Index<0:0:2>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
    %20 = VPUMI37XX.KernelParams inputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : vector<80xui8>) -> !VPURegMapped.Index<0:0:1>
    %21 = VPUMI37XX.DeclareKernelText kernel_path("activation_sigmoid") -> !VPURegMapped.Index<0:0:2>
    %22 = VPUMI37XX.DeclareKernelArgs kernel_path("activation_sigmoid") -> !VPURegMapped.Index<0:0:2>
    %23 = VPUMI37XX.DeclareKernelEntry kernel_path("activation_sigmoid") -> !VPURegMapped.Index<0:0:2>
    %24 = VPUMI37XX.ActKernelRange kernel_text_index(%21 : !VPURegMapped.Index<0:0:2>) kernel_args_index(%22 : !VPURegMapped.Index<0:0:2>) kernel_entry_index(%23 : !VPURegMapped.Index<0:0:2>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:2>
    %25 = VPUMI37XX.ActKernelInvocation range_index(%24 : <0:0:2>) waits(%6 : !VPURegMapped.Index<0:0:2>) updates(%7 : !VPURegMapped.Index<0:0:3>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>
    %26 = VPUMI37XX.KernelParams inputs(%2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_sigmoid") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPURegMapped.Index<0:0:2>
    return %arg1 : memref<1x1x1x1000xf16>

    // CHECK:      ELFNPU37XX.CreateSection {{.+}}dmaTasks
        // CHECK:      ELFNPU37XX.PutOpInSection
        // CHECK-NOT:      ELFNPU37XX.Pad
        // CHECK:      ELFNPU37XX.PutOpInSection
        // CHECK-NOT:      ELFNPU37XX.Pad

    // CHECK:      ELFNPU37XX.CreateSection {{.+}}BarrierConfigs
        // CHECK:      ELFNPU37XX.PutOpInSection
        // CHECK-NOT:      ELFNPU37XX.Pad
        // CHECK:      ELFNPU37XX.PutOpInSection
        // CHECK-NOT:      ELFNPU37XX.Pad
        // CHECK:      ELFNPU37XX.PutOpInSection
        // CHECK-NOT:      ELFNPU37XX.Pad
        // CHECK:      ELFNPU37XX.PutOpInSection
        // CHECK-NOT:      ELFNPU37XX.Pad

    // CHECK:      ELFNPU37XX.CreateSection {{.+}}KernelText
        // CHECK:      ELFNPU37XX.PutOpInSection
        // CHECK:      ELFNPU37XX.Pad
        // CHECK:      ELFNPU37XX.PutOpInSection
        // CHECK:      ELFNPU37XX.Pad
        // CHECK:      ELFNPU37XX.PutOpInSection
        // CHECK-NOT:      ELFNPU37XX.Pad

    // CHECK:      ELFNPU37XX.CreateSection {{.+}}KernelData
        // CHECK:      ELFNPU37XX.PutOpInSection
        // CHECK-NOT:      ELFNPU37XX.Pad
        // CHECK:      ELFNPU37XX.PutOpInSection
        // CHECK-NOT:      ELFNPU37XX.Pad
        // CHECK:      ELFNPU37XX.PutOpInSection
        // CHECK-NOT:      ELFNPU37XX.Pad

    // CHECK:      ELFNPU37XX.CreateSection {{.+}}KernelParams
        // CHECK:      ELFNPU37XX.PutOpInSection
        // CHECK:      ELFNPU37XX.Pad
        // CHECK:      ELFNPU37XX.PutOpInSection
        // CHECK:      ELFNPU37XX.Pad
        // CHECK:      ELFNPU37XX.PutOpInSection
        // CHECK-NOT:      ELFNPU37XX.Pad

    // CHECK:      ELFNPU37XX.CreateSection {{.+}}ActKernelRanges
        // CHECK:      ELFNPU37XX.PutOpInSection
        // CHECK-NOT:      ELFNPU37XX.Pad
        // CHECK:      ELFNPU37XX.PutOpInSection
        // CHECK-NOT:      ELFNPU37XX.Pad
        // CHECK:      ELFNPU37XX.PutOpInSection
        // CHECK-NOT:      ELFNPU37XX.Pad

    // CHECK:      ELFNPU37XX.CreateSection {{.+}}ActKernelInvocations
        // CHECK:      ELFNPU37XX.PutOpInSection
        // CHECK-NOT:      ELFNPU37XX.Pad
        // CHECK:      ELFNPU37XX.PutOpInSection
        // CHECK-NOT:      ELFNPU37XX.Pad
        // CHECK:      ELFNPU37XX.PutOpInSection
        // CHECK-NOT:      ELFNPU37XX.Pad

    // CHECK:      ELFNPU37XX.CreateSection {{.+}}MappedInference
        // CHECK:      ELFNPU37XX.PutOpInSection
        // CHECK-NOT:      ELFNPU37XX.Pad

    // CHECK:      ELFNPU37XX.CreateMetadataSection
        // CHECK:      VPUMI37XX.NetworkMetadata
        // CHECK-NOT:      ELFNPU37XX.Pad
  }
}
