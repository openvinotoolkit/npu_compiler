//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch% allow-custom-values=true" --convert-VPUMI37XX-to-ELF %s | FileCheck %s
// REQUIRES: arch-NPU37XX

module @Test attributes {config.arch = #config.arch_kind<NPU37XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
  config.MemoryResource 31457280 bytes of @DDR {config.bandwidth = 8 : i64, config.derateFactor = 6.000000e-01 : f64}
  config.ExecutorResource 1 of @DMA_NN
  config.Resources 1 of @NCE {
    config.MemoryResource 2097152 bytes of @CMX_NN {config.bandwidth = 32 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 1 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1000xf16>
  } outputsInfo : {
    DataInfo "hswish" : tensor<1x1000xf16>
  }
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
  module @VPU.SW {
    func.func private @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_hswish.cpp", VPU.kernel_entry = "activation_hswish"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }
  func.func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %2 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    %3 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>
    %5 = VPUMI37XX.DeclareKernelText kernel_path("activation_hswish_3720xx") -> !VPURegMapped.Index<0:0:0>
    %6 = VPUMI37XX.DeclareKernelArgs kernel_path("activation_hswish_3720xx") -> !VPURegMapped.Index<0:0:0>
    %7 = VPUMI37XX.DeclareKernelEntry kernel_path("activation_hswish_3720xx") -> !VPURegMapped.Index<0:0:0>
    %8 = VPUMI37XX.ActKernelRange kernel_text_index(%5 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%7 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
    %10 = VPUMI37XX.KernelParams inputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPURegMapped.Index<0:0:0>
    %9 = VPUMI37XX.ActKernelInvocation range_index(%8 : <0:0:0>) params_index(%10 : !VPURegMapped.Index<0:0:0>) waits(%2 : !VPURegMapped.Index<0:0:0>) updates(%3 : !VPURegMapped.Index<0:0:1>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %12 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %13 = VPURT.DeclareBuffer <CMX_NN> [0] <4000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %14 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:2>
    %15 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:3>
    %17 = VPUMI37XX.DeclareKernelText kernel_path("activation_hswish_3720xx") -> !VPURegMapped.Index<0:0:1>
    %18 = VPUMI37XX.DeclareKernelArgs kernel_path("activation_hswish_3720xx") -> !VPURegMapped.Index<0:0:1>
    %19 = VPUMI37XX.DeclareKernelEntry kernel_path("activation_hswish_3720xx") -> !VPURegMapped.Index<0:0:1>
    %20 = VPUMI37XX.ActKernelRange kernel_text_index(%17 : !VPURegMapped.Index<0:0:1>) kernel_args_index(%18 : !VPURegMapped.Index<0:0:1>) kernel_entry_index(%19 : !VPURegMapped.Index<0:0:1>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:1>
    %22 = VPUMI37XX.KernelParams inputs(%12 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%13 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPURegMapped.Index<0:0:1>
    %21 = VPUMI37XX.ActKernelInvocation range_index(%20 : <0:0:1>) params_index(%22 : !VPURegMapped.Index<0:0:1>) waits(%14 : !VPURegMapped.Index<0:0:2>) updates(%15 : !VPURegMapped.Index<0:0:3>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
    %24 = VPURT.DeclareBuffer <CMX_NN> [0] <6000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %25 = VPURT.DeclareBuffer <CMX_NN> [0] <8000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %26 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:4>
    %27 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:5>
    %29 = VPUMI37XX.DeclareKernelText kernel_path("activation_hswish_3720xx") -> !VPURegMapped.Index<0:0:2>
    %30 = VPUMI37XX.DeclareKernelArgs kernel_path("activation_hswish_3720xx") -> !VPURegMapped.Index<0:0:2>
    %31 = VPUMI37XX.DeclareKernelEntry kernel_path("activation_hswish_3720xx") -> !VPURegMapped.Index<0:0:2>
    %32 = VPUMI37XX.ActKernelRange kernel_text_index(%29 : !VPURegMapped.Index<0:0:2>) kernel_args_index(%30 : !VPURegMapped.Index<0:0:2>) kernel_entry_index(%31 : !VPURegMapped.Index<0:0:2>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:2>
    %34 = VPUMI37XX.KernelParams inputs(%24 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%25 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPURegMapped.Index<0:0:2>
    %33 = VPUMI37XX.ActKernelInvocation range_index(%32 : <0:0:2>) params_index(%34 : !VPURegMapped.Index<0:0:2>) waits(%26 : !VPURegMapped.Index<0:0:4>) updates(%27 : !VPURegMapped.Index<0:0:5>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>
    %36 = VPUMI37XX.MappedInference actKernelRanges(%8 : !VPURegMapped.Index<0:0:0>) actKernelInvocations(%9 : !VPURegMapped.Index<0:0:0>) barriers(%2 : !VPURegMapped.Index<0:0:0>) dmaCount([0]) invariantCount(0) variantCount(0) actKernelRangesCount(3) actKernelInvocationsCount(3) barrierCount(6) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x1x1x1000xf16>
  }
}

//CHECK: %[[VAL0:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL1:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL2:.*]] = VPUMI37XX.ConfigureBarrier
//CHECK: %[[VAL3:.*]] = VPUMI37XX.ConfigureBarrier
//CHECK: %[[VAL5:.*]] = VPUMI37XX.DeclareKernelText
//CHECK: %[[VAL6:.*]] = VPUMI37XX.DeclareKernelArgs
//CHECK: %[[VAL7:.*]] = VPUMI37XX.DeclareKernelEntry
//CHECK: %[[VAL8:.*]] = VPUMI37XX.ActKernelRange
//CHECK: %[[VAL10:.*]] = VPUMI37XX.KernelParams
//CHECK: %[[VAL9:.*]] = VPUMI37XX.ActKernelInvocation

//CHECK: %[[VAL12:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL13:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL14:.*]] = VPUMI37XX.ConfigureBarrier
//CHECK: %[[VAL15:.*]] = VPUMI37XX.ConfigureBarrier
//CHECK: %[[VAL17:.*]] = VPUMI37XX.DeclareKernelText
//CHECK: %[[VAL18:.*]] = VPUMI37XX.DeclareKernelArgs
//CHECK: %[[VAL19:.*]] = VPUMI37XX.DeclareKernelEntry
//CHECK: %[[VAL20:.*]] = VPUMI37XX.ActKernelRange
//CHECK: %[[VAL22:.*]] = VPUMI37XX.KernelParams
//CHECK: %[[VAL21:.*]] = VPUMI37XX.ActKernelInvocation

//CHECK: %[[VAL24:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL25:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL26:.*]] = VPUMI37XX.ConfigureBarrier
//CHECK: %[[VAL27:.*]] = VPUMI37XX.ConfigureBarrier
//CHECK: %[[VAL29:.*]] = VPUMI37XX.DeclareKernelText
//CHECK: %[[VAL30:.*]] = VPUMI37XX.DeclareKernelArgs
//CHECK: %[[VAL31:.*]] = VPUMI37XX.DeclareKernelEntry
//CHECK: %[[VAL32:.*]] = VPUMI37XX.ActKernelRange
//CHECK: %[[VAL34:.*]] = VPUMI37XX.KernelParams
//CHECK: %[[VAL33:.*]] = VPUMI37XX.ActKernelInvocation

//CHECK-DAG: %[[VAL48:.*]] = VPUMI37XX.ActShaveRt kernel("nnActEntry") -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: %[[VAL49:.*]] = ELFNPU37XX.CreateSection {{.*}} {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".text.actKernelRtConfigSec"} -> !ELFNPU37XX.Section {
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL48]] : !VPURegMapped.Index<0:0:0>
//CHECK: %[[VAL50:.*]]  = ELFNPU37XX.Symbol %[[VAL49]] name("sym_actKernelRtConfigsSec") : !ELFNPU37XX.Section

//CHECK-DAG: %[[VAL51:.*]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.actKernelRtConfig")
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL50]] : !ELFNPU37XX.Symbol

//CHECK-DAG: %[[KERNELTEXT:.*]] = ELFNPU37XX.CreateSection {{.*}} secName = ".text.KernelText"
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL5]] : !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: ELFNPU37XX.Pad {{.*}}
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL17]] : !VPURegMapped.Index<0:0:1>
//CHECK-NEXT: ELFNPU37XX.Pad {{.*}}
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL29]] : !VPURegMapped.Index<0:0:2>
//CHECK-NEXT: ELFNPU37XX.Pad size(128)

//CHECK-DAG: %[[KERNELDATA:.*]] = ELFNPU37XX.CreateSection {{.*}} secName = ".text.KernelData"
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL6]] : !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL18]] : !VPURegMapped.Index<0:0:1>
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL30]] : !VPURegMapped.Index<0:0:2>

//CHECK-DAG: %[[KERNELPARAMS:.*]] = ELFNPU37XX.CreateSection {{.*}} secName = ".text.KernelParams"
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL10]] : !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: ELFNPU37XX.Pad {{.*}}
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL22]] : !VPURegMapped.Index<0:0:1>
//CHECK-NEXT: ELFNPU37XX.Pad {{.*}}
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL34]] : !VPURegMapped.Index<0:0:2>

//CHECK-DAG: %[[ACTKERNELR:.*]] = ELFNPU37XX.CreateSection {{.*}} secName = ".text.ActKernelRanges"
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL8]] : !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL20]] : !VPURegMapped.Index<0:0:1>
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL32]] : !VPURegMapped.Index<0:0:2>

//CHECK-DAG: %[[ACTKERNELI:.*]] = ELFNPU37XX.CreateSection {{.*}} secName = ".text.ActKernelInvocations"
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL9]] : !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL21]] : !VPURegMapped.Index<0:0:1>
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL33]] : !VPURegMapped.Index<0:0:2>

//CHECK: %[[BUILTIN_SYMTABSEC:.*]] = ELFNPU37XX.CreateSymbolTableSection secName("VPU_RT_SYMTAB")
//CHECK: %[[SYMTABSEC:.*]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.tasks")

//CHECK: ELFNPU37XX.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection(%[[SYMTABSEC]])
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL10]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL10]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL10]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL10]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL22]] : !VPURegMapped.Index<0:0:1>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL22]] : !VPURegMapped.Index<0:0:1>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL22]] : !VPURegMapped.Index<0:0:1>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL22]] : !VPURegMapped.Index<0:0:1>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL34]] : !VPURegMapped.Index<0:0:2>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL34]] : !VPURegMapped.Index<0:0:2>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL34]] : !VPURegMapped.Index<0:0:2>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL34]] : !VPURegMapped.Index<0:0:2>) {{.*}} <R_VPU_32>

//CHECK: ELFNPU37XX.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection(%[[BUILTIN_SYMTABSEC]])
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL10]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL10]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL22]] : !VPURegMapped.Index<0:0:1>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL22]] : !VPURegMapped.Index<0:0:1>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL34]] : !VPURegMapped.Index<0:0:2>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL34]] : !VPURegMapped.Index<0:0:2>) {{.*}} <R_VPU_32>

//CHECK: ELFNPU37XX.CreateRelocationSection secName(".rlt.text.ActKernelRanges") sourceSymbolTableSection(%[[SYMTABSEC]])
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL8]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL20]] : !VPURegMapped.Index<0:0:1>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL32]] : !VPURegMapped.Index<0:0:2>) {{.*}} <R_VPU_32>

//CHECK: ELFNPU37XX.CreateRelocationSection secName(".rlt.text.ActKernelInvocations") sourceSymbolTableSection(%[[BUILTIN_SYMTABSEC]])
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL9]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32_RTM>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL21]] : !VPURegMapped.Index<0:0:1>) {{.*}} <R_VPU_32_RTM>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL33]] : !VPURegMapped.Index<0:0:2>) {{.*}} <R_VPU_32_RTM>

//CHECK: ELFNPU37XX.CreateRelocationSection secName(".rlt.text.ActKernelInvocations") sourceSymbolTableSection(%[[SYMTABSEC]])
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL9]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL9]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL21]] : !VPURegMapped.Index<0:0:1>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL21]] : !VPURegMapped.Index<0:0:1>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL33]] : !VPURegMapped.Index<0:0:2>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.Reloc baseOp(%[[VAL33]] : !VPURegMapped.Index<0:0:2>) {{.*}} <R_VPU_32>
