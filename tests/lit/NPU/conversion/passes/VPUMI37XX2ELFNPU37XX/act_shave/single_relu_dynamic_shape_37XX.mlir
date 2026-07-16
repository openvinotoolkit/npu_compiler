//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform% allow-custom-values=true" --convert-VPUMI37XX-to-ELF %s | FileCheck %s
// REQUIRES: platform-NPU3720

module @SimpleActivation attributes {config.compilationMode = #config.compilation_mode<ReferenceSW>} {
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_ReLU(memref<2x4x20x20xf16, [@CMX_NN, 0]>, memref<2x4x20x20xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Param_1" : tensor<?x?x?x?xf16>
    DataInfo "vpux_ie_shape_Param_1" : tensor<4xi32>
  } outputsInfo : {
    DataInfo "Relu_2" : tensor<?x?x?x?xf16>
    DataInfo "vpux_ie_shape_Relu_2" : tensor<4xi32>
  }
  func.func @main(%arg0: memref<2x4x20x20xf16, @DDR>, %arg1: memref<4xi32, @DDR>, %arg2: memref<2x4x20x20xf16, @DDR>, %arg3: memref<4xi32, @DDR>) -> (memref<2x4x20x20xf16, @DDR>, memref<4xi32, @DDR>) {
    %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<2x4x20x20xf16, @DDR>
    %1 = VPURT.DeclareBuffer <NetworkInput> [1] <0> -> memref<4xi32, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<2x4x20x20xf16, @DDR>
    %3 = VPURT.DeclareBuffer <NetworkOutput> [1] <0> -> memref<4xi32, @DDR>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<2x4x20x20xf16, [@CMX_NN, 0]>
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <6400> -> memref<4xi32, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <6464> -> memref<2x4x20x20xf16, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <12864> -> memref<4xi32, [@CMX_NN, 0]>
    %8 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    %9 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>
    %10 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<2, -1> -> !VPURegMapped.Index<0:0:2>
    %11 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<3, -1> -> !VPURegMapped.Index<0:0:3>
    %12 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<4, -1> -> !VPURegMapped.Index<0:0:4>
    %15 = VPUMI37XX.DeclareKernelText kernel_path("activation_relu_3720xx") -> !VPURegMapped.Index<0:0:0>
    %16 = VPUMI37XX.DeclareKernelArgs kernel_path("activation_relu_3720xx") -> !VPURegMapped.Index<0:0:0>
    %17 = VPUMI37XX.DeclareKernelEntry kernel_path("activation_relu_3720xx") -> !VPURegMapped.Index<0:0:0>
    %18 = VPUMI37XX.ActKernelRange kernel_text_index(%15 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%16 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%17 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
    %20 = VPUMI37XX.KernelParams inputs(%4 : memref<2x4x20x20xf16, [@CMX_NN, 0]>) outputs(%6 : memref<2x4x20x20xf16, [@CMX_NN, 0]>) dynamicInputShapes((%5) : (memref<4xi32, [@CMX_NN, 0]>)) dynamicOutputShapes((%7) : (memref<4xi32, [@CMX_NN, 0]>)) kernel_type("activation_relu") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPURegMapped.Index<0:0:0>
    %19 = VPUMI37XX.ActKernelInvocation range_index(%18 : <0:0:0>) params_index(%20 : !VPURegMapped.Index<0:0:0>) waits(%9 : !VPURegMapped.Index<0:0:1>) updates(%10 : !VPURegMapped.Index<0:0:2>) tile(0) start_after(3) clean_after(2) -> !VPURegMapped.Index<0:0:0>
    %23 = VPUMI37XX.MappedInference actKernelRanges(%18 : !VPURegMapped.Index<0:0:0>) actKernelInvocations(%19 : !VPURegMapped.Index<0:0:0>) barriers(%8 : !VPURegMapped.Index<0:0:0>) dmaCount([0, 0]) invariantCount(0) variantCount(0) actKernelRangesCount(1) actKernelInvocationsCount(1) barrierCount(5) -> !VPURegMapped.Index<0:0:0>
    return %arg2, %arg3 : memref<2x4x20x20xf16, @DDR>, memref<4xi32, @DDR>
  }
}

//CHECK: [[VAL0:%.+]] = VPURT.DeclareBuffer
//CHECK: [[VAL1:%.+]] = VPURT.DeclareBuffer
//CHECK: [[VAL2:%.+]] = VPURT.DeclareBuffer
//CHECK: [[VAL3:%.+]] = VPURT.DeclareBuffer
//CHECK: [[VAL4:%.+]] = VPURT.DeclareBuffer
//CHECK: [[VAL5:%.+]] = VPURT.DeclareBuffer
//CHECK: [[VAL6:%.+]] = VPUMI37XX.ConfigureBarrier
//CHECK: [[VAL7:%.+]] = VPUMI37XX.ConfigureBarrier
//CHECK: [[VAL8:%.+]] = VPUMI37XX.ConfigureBarrier
//CHECK: [[VAL9:%.+]] = VPUMI37XX.ConfigureBarrier
//CHECK: [[VAL10:%.+]] = VPUMI37XX.ConfigureBarrier
//CHECK: [[VAL13:%.+]] = VPUMI37XX.DeclareKernelText
//CHECK: [[VAL14:%.+]] = VPUMI37XX.DeclareKernelArgs
//CHECK: [[VAL15:%.+]] = VPUMI37XX.DeclareKernelEntry
//CHECK: [[VAL16:%.+]] = VPUMI37XX.ActKernelRange
//CHECK: [[VAL18:%.+]] = VPUMI37XX.KernelParams
//CHECK: [[VAL17:%.+]] = VPUMI37XX.ActKernelInvocation

//CHECK: [[BUILTIN_SYMTABSEC:%.+]] = ELFNPU37XX.CreateSymbolTableSection secName("VPU_RT_SYMTAB")
//CHECK: [[SYMTABSEC:%.+]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.tasks")

//CHECK-DAG: ELFNPU37XX.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection([[SYMTABSEC]])
//CHECK-NEXT: ELFNPU37XX.Reloc baseOp([[VAL18]] : !VPURegMapped.Index<0:0:0>) {{.+}} <R_VPU_32>
//CHECK-NEXT: ELFNPU37XX.Reloc baseOp([[VAL18]] : !VPURegMapped.Index<0:0:0>) {{.+}} <R_VPU_32>

//CHECK-DAG: ELFNPU37XX.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection([[BUILTIN_SYMTABSEC]])
//CHECK-NEXT: ELFNPU37XX.Reloc baseOp([[VAL18]] : !VPURegMapped.Index<0:0:0>) {{.+}} <R_VPU_32>
//CHECK-NEXT: ELFNPU37XX.Reloc baseOp([[VAL18]] : !VPURegMapped.Index<0:0:0>) {{.+}} <R_VPU_32>
//CHECK-NEXT: ELFNPU37XX.Reloc baseOp([[VAL18]] : !VPURegMapped.Index<0:0:0>) {{.+}} <R_VPU_32>
//CHECK-NEXT: ELFNPU37XX.Reloc baseOp([[VAL18]] : !VPURegMapped.Index<0:0:0>) {{.+}} <R_VPU_32>
