//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --reorder-mapped-inference-ops %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @test attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
  config.Resources {activity_factor = 0.092296911622323521 : f64} 4 of @NCE at 1.700000e+03 MHz {
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
  func.func @main() {
    %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<12xf16, @DDR>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<12xf16, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1xf16>
    %3 = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<1xui32>
    %4 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1xf16>
    %5 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%4 : memref<1xf16>) outputs(%2 : memref<1xf16>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %6 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <0, -1> -> !VPURegMapped.Index<0:0:0>
    %7 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <1, -1> -> !VPURegMapped.Index<0:0:1>
    %8 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
    %9 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:1>
    %10 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <2, -1> -> !VPURegMapped.Index<0:0:2>
    %11 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%4 : memref<1xf16>) outputs(%2 : memref<1xf16>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:0:0>
    %12 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <3, -1> -> !VPURegMapped.Index<0:0:3>
    %13 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<1:0:0>
    %14 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<1:0:1>
    %15 = VPUMI40XX.DeclareKernelEntry kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
    %16 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
    %17 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
    %18 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
    %19 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<1:0:0>
    %20 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<1:0:1>
    %21 = VPUMI40XX.DeclareKernelText kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
    %22 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
    %23 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
    %24 = VPUMI40XX.ActKernelRange kernel_text_index(%21 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%18 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%15 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
    %25 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<1:0:0>
    %26 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<1:0:1>
    %27 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%4 : memref<1xf16>) outputs(%2 : memref<1xf16>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %28 = VPUMI40XX.KernelParams inputs(%4 : memref<1xf16>) outputs(%2 : memref<1xf16>) kernel_type("activation_softmax") kernel_params([0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]) -> !VPURegMapped.Index<0:0:0>
    %29 = VPUMI40XX.ActKernelInvocation range_index(%24 : <0:0:0>) kernel_params(%28 : <0:0:0>) tile(7) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %30 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%4 : memref<1xf16>) outputs(%2 : memref<1xf16>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>

    return
  }
}


//CHECK:      VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
//CHECK-NEXT: VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
//CHECK-NEXT: VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:1>
//CHECK-NEXT: VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<1:0:0>
//CHECK-NEXT: VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<1:0:1>
//CHECK-NEXT: VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<1:0:0>
//CHECK-NEXT: VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<1:0:1>
//CHECK-NEXT: VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<1:0:0>
//CHECK-NEXT: VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<1:0:1>
//CHECK-NEXT: VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1xf16>
//CHECK-NEXT: VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1xf16>
//CHECK-NEXT: VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<1xui32>
//CHECK-NEXT: VPURT.DeclareBuffer <DDR> <0> -> memref<12xf16, @DDR>
//CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<12xf16, [@CMX_NN, 0]>
//CHECK-NEXT: VPUMI40XX.DeclareKernelText kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: VPUMI40XX.DeclareKernelEntry kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: VPUMI40XX.KernelParams
//CHECK-NEXT: VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <0, -1> -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <1, -1> -> !VPURegMapped.Index<0:0:1>
//CHECK-NEXT: VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <2, -1> -> !VPURegMapped.Index<0:0:2>
//CHECK-NEXT: VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <3, -1> -> !VPURegMapped.Index<0:0:3>
//CHECK-NEXT: VPUMI40XX.ActKernelRange kernel_text_index(%17 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%19 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%18 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: VPUMI40XX.ActKernelInvocation range_index(%25 : <0:0:0>) kernel_params(%20 : <0:0:0>) tile(7) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: VPUMI40XX.NNDMA {port = 0 : i64} inputs(%12 : memref<1xf16>) outputs(%13 : memref<1xf16>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: VPUMI40XX.NNDMA {port = 0 : i64} inputs(%12 : memref<1xf16>) outputs(%13 : memref<1xf16>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
//CHECK-NEXT: VPUMI40XX.NNDMA {port = 1 : i64} inputs(%12 : memref<1xf16>) outputs(%13 : memref<1xf16>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
//CHECK-NEXT: VPUMI40XX.NNDMA {port = 1 : i64} inputs(%12 : memref<1xf16>) outputs(%13 : memref<1xf16>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:0:0>
