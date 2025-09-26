//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --convert-VPUMI37XX-to-ELF %s | FileCheck %s
// REQUIRES: arch-NPU37XX

module attributes {config.arch = #config.arch_kind<NPU37XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
  config.Resources 2 of @NCE at 1.300000e+03 MHz {
    config.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1982464 bytes of @CMX_NN {config.bandwidth = 32 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @SHAVE_NN
    config.ExecutorResource 1 of @DPU
  }
  config.ExecutorResource 2 of @DMA_NN
  config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 8 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo entryPoint : @SWKernelDynamicInputs inputsInfo : {
    DataInfo "input_bound" : tensor<1x3x10x10xf16>
    DataInfo "input_shape" : tensor<4xsi32>
  } outputsInfo : {
    DataInfo "output_bound" : tensor<1x3x10x10xf16>
    DataInfo "output_shape" : tensor<4xsi32>
  }
  module @VPU.SW {
    func.func private @builtin_dummy(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "dummy.cpp", VPU.kernel_entry = "dummy"}
  }
  // CHECK-LABEL: @SWKernelDynamicInputs
  func.func @SWKernelDynamicInputs(%arg0: memref<1x3x10x10xf16>, %arg1: memref<4xsi32>) -> (memref<1x3x10x10xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>) {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x10x10xf16, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <600> -> memref<4xsi32, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <616> -> memref<1x3x10x10xf16, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <1216> -> memref<4xsi32, [@CMX_NN, 0]>
    %4 = VPUMI37XX.DeclareKernelText kernel_path("dummy_3720xx") -> !VPURegMapped.Index<0:0:0>
    %5 = VPUMI37XX.DeclareKernelEntry kernel_path("dummy_3720xx") -> !VPURegMapped.Index<0:0:0>
    %6 = VPUMI37XX.DeclareKernelArgs kernel_path("dummy_3720xx") -> !VPURegMapped.Index<0:0:0>
    %7 = VPUMI37XX.ActKernelRange kernel_text_index(%4 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%6 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%5 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
    %8 = VPUMI37XX.KernelParams inputs(%0 : memref<1x3x10x10xf16, [@CMX_NN, 0]>) outputs(%2 : memref<1x3x10x10xf16, [@CMX_NN, 0]>) dynamicInputShapes((%1) : (memref<4xsi32, [@CMX_NN, 0]>)) dynamicOutputShapes((%3) : (memref<4xsi32, [@CMX_NN, 0]>)) kernel_type("dummy") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPURegMapped.Index<0:0:0>
    %9 = VPUMI37XX.ActKernelInvocation range_index(%7 : <0:0:0>) params_index(%8 : !VPURegMapped.Index<0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %10 = VPUMI37XX.MappedInference actKernelRanges(%7 : !VPURegMapped.Index<0:0:0>) actKernelInvocations(%9 : !VPURegMapped.Index<0:0:0>) dmaCount([0, 0]) invariantCount(0) variantCount(0) actKernelRangesCount(1) actKernelInvocationsCount(1) barrierCount(0) -> !VPURegMapped.Index<0:0:0>
    return %2, %3 : memref<1x3x10x10xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>
  }
  ///

  //CHECK:        [[IN_BOUND:%.*]] = VPURT.DeclareBuffer
  //CHECK:        [[IN_DYN_SHAPE:%.*]] = VPURT.DeclareBuffer
  //CHECK:        [[OUT_BOUND:%.*]] = VPURT.DeclareBuffer
  //CHECK:        [[OUT_DYN_SHAPE:%.*]] = VPURT.DeclareBuffer
  //CHECK:        %[[VAL1:.*]] = VPUMI37XX.DeclareKernelText
  //CHECK:        %[[VAL2:.*]] = VPUMI37XX.DeclareKernelEntry
  //CHECK:        %[[VAL3:.*]] = VPUMI37XX.DeclareKernelArgs
  //CHECK:        %[[VAL4:.*]] = VPUMI37XX.ActKernelRange

  //CHECK:        %[[VAL5:.*]] = VPUMI37XX.KernelParams
  //CHECK-SAME:   inputs([[IN_BOUND]]
  //CHECK-SAME:   outputs([[OUT_BOUND]]
  //CHECK-SAME:   dynamicInputShapes(([[IN_DYN_SHAPE]])
  //CHECK-SAME:   dynamicOutputShapes(([[OUT_DYN_SHAPE]])

  //CHECK:        %[[VAL6:.*]] = VPUMI37XX.ActKernelInvocation
}
