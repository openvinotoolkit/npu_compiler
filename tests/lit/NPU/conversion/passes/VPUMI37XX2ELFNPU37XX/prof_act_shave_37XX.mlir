//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform%" --lower-VPUIP-to-ELF %s | FileCheck %s
// REQUIRES: platform-NPU3720

module @actShaveProfiling {
  net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  } profilingOutputsInfo : {
    DataInfo "profilingOutput" {
      VPUIP.ProfilingSection type 3 : 16 bytes from 0
    } : tensor<4xui32>
  }

  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_hswish.cpp", VPU.kernel_entry = "activation_hswish"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

  // CHECK-LABEL: func.func @main
  // CHECK-SAME:    [[ARG_0:%[^:]+]]: memref<1x1x1x1000xf16, @DDR>
  // CHECK-SAME:    [[ARG_1:%[^:]+]]: memref<1x1x1x1000xf16, @DDR>
  // CHECK-SAME:    [[ARG_2:%[^:]+]]: memref<4xui32>
  func.func @main(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>, %arg2: memref<4xui32>) -> (memref<1x1x1x1000xf16, @DDR>, memref<4xui32>) {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    %prof_buf_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <4000> -> memref<4xui32, [@CMX_NN, 0]>
    %prof_output = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<4xui32, @DDR>

    %5 = VPUMI37XX.DeclareKernelText kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
    %6 = VPUMI37XX.DeclareKernelArgs kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
    %7 = VPUMI37XX.DeclareKernelEntry kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
    %8 = VPUMI37XX.ActKernelRange kernel_text_index(%5 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%7 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
    %10 = VPUMI37XX.KernelParams inputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPURegMapped.Index<0:0:0>
    %9 = VPUMI37XX.ActKernelInvocation range_index(%8 : <0:0:0>) params_index(%10 : !VPURegMapped.Index<0:0:0>) profiling_data(%prof_buf_cmx : memref<4xui32, [@CMX_NN, 0]>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>

    %prof_dma = VPUMI37XX.NNDMA <{port = 0 : i64}> inputs(%prof_buf_cmx : memref<4xui32, [@CMX_NN, 0]>) outputs(%prof_output : memref<4xui32, @DDR>) start_after(0) clean_after(6) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

    %13 = VPUMI37XX.MappedInference dmas(%prof_dma : !VPURegMapped.Index<0:0:0>) actKernelRanges(%8 : !VPURegMapped.Index<0:0:0>) actKernelInvocations(%9 : !VPURegMapped.Index<0:0:0>) dmaCount([1, 0]) invariantCount(0) variantCount(0) actKernelRangesCount(1) actKernelInvocationsCount(1) barrierCount(0) -> !VPURegMapped.Index<0:0:0>
    return %arg1, %arg2 : memref<1x1x1x1000xf16, @DDR>, memref<4xui32>

    // CHECK:       [[BUF_IN:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:       [[BUF_OUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:       [[PROF_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4000> -> memref<4xui32, [@CMX_NN, 0]>
    // CHECK:       [[PROF_OUT:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<4xui32, @DDR>

    // CHECK:       [[KERNEL_TEXT:%.+]] = VPUMI37XX.DeclareKernelText
    // CHECK:       [[KERNEL_ARGS:%.+]] = VPUMI37XX.DeclareKernelArgs
    // CHECK:       [[KERNEL_ENTRY:%.+]] = VPUMI37XX.DeclareKernelEntry
    // CHECK:       [[KERNEL_RANGE:%.+]] = VPUMI37XX.ActKernelRange
    // CHECK:       [[KERNEL_PARAMS:%.+]] = VPUMI37XX.KernelParams
    // CHECK:       [[KERNEL_INVO:%.+]] = VPUMI37XX.ActKernelInvocation
    // CHECK-SAME:       profiling_data([[PROF_BUF]] : memref<4xui32, [@CMX_NN, 0]>)

    // CHECK:       [[DMA_PROF_TO_OUT:%.+]] = VPUMI37XX.NNDMA
    // CHECK-SAME:       inputs([[PROF_BUF]] : memref<4xui32, [@CMX_NN, 0]>)
    // CHECK-SAME:       outputs([[PROF_OUT]] : memref<4xui32, @DDR>)

    // CHECK:       [[SYM_IN:%.+]] = ELFNPU37XX.Symbol [[ARG_0]] name("input") size(2000) : memref<1x1x1x1000xf16, @DDR>
    // CHECK:       [[SYM_OUT:%.+]] = ELFNPU37XX.Symbol [[ARG_1]] name("output") size(2000) : memref<1x1x1x1000xf16, @DDR>
    // CHECK:       [[SYM_PROF:%.+]] = ELFNPU37XX.Symbol [[ARG_2]] name("profilingOutput") size(16) : memref<4xui32>


    // CHECK:       [[VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR:%.+]] = ELFNPU37XX.Symbol {{%.+}} name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") <{isBuiltin}> : i8
    // CHECK:       [[VPU_NNRD_SYM_RTM_ACT:%.+]] = ELFNPU37XX.Symbol {{%.+}} name("VPU_NNRD_SYM_RTM_ACT") <{isBuiltin}> : i8
    // CHECK:       [[VPU_RT_SYMTAB:%.+]] = ELFNPU37XX.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags("SHF_NONE") <{isBuiltin}> -> !ELFNPU37XX.Section {

    // CHECK:       [[RELOC_KERNEL_INVO:%.+]] = ELFNPU37XX.CreateRelocationSection secName(".rlt.text.ActKernelInvocations") sourceSymbolTableSection([[VPU_RT_SYMTAB]])
    // CHECK-NEXT:       ELFNPU37XX.Reloc baseOp([[KERNEL_INVO]] : !VPURegMapped.Index<0:0:0>) offset({{.+}}) <R_VPU_32_RTM> [[VPU_NNRD_SYM_RTM_ACT]] 24
    // CHECK-NEXT:       ELFNPU37XX.Reloc baseOp([[KERNEL_INVO]] : !VPURegMapped.Index<0:0:0>) offset(12) <R_VPU_32_SUM> [[VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR]] 4000
    // CHECK-NEXT:       ELFNPU37XX.Reloc baseOp([[KERNEL_INVO]] : !VPURegMapped.Index<0:0:0>) offset(16) <R_VPU_64_LSHIFT>
    // CHECK-NEXT:       ELFNPU37XX.Reloc baseOp([[KERNEL_INVO]] : !VPURegMapped.Index<0:0:0>) offset(24) <R_VPU_64_LSHIFT>
    // CHECK-NEXT:       ELFNPU37XX.Reloc baseOp([[KERNEL_INVO]] : !VPURegMapped.Index<0:0:0>) offset(52) <R_VPU_32>
  }
}
