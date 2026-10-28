//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --cmx-stack-frames-reserve-mem --cmx-metadata-reserve-mem --add-shave-dpu-descriptors %s | FileCheck %s
// REQUIRES: platform-NPU5010

// A DPU parameters buffer is written to every tile when the network has a SHAVE task using the DPU

module @ShaveDpuDescriptorsWithDpuShaveTask attributes {config.compilationMode = #config.compilation_mode<DefaultHW>, config.revisionID = #config.revision_id<REVISION_NONE>} {
  config.Resources 1 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.ExecutorResource 1 of @M2I
  config.ExecutorResource 1 of @DMA_NN
  config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x16x16xf16>
  }
  func.func @main(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<1x16x16x16xf16, @DDR>) -> memref<1x16x16x16xf16, @DDR> {
    %in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x16x16xf16, @DDR>
    %out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x16x16xf16, @DDR>
    %cmx = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, [@CMX_NN, 0]>

    %bar = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}> <0, -1> -> !VPURegMapped.Index<0:0:0>

    %dma0 = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs(%in : memref<1x16x16x16xf16, @DDR>) outputs(%cmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) updates(%bar : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %dma1 = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs(%cmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) outputs(%out : memref<1x16x16x16xf16, @DDR>) previousDMA(%dma0 : !VPURegMapped.Index<0:0:0>) waits(%bar : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>

    %entry = VPUMI40XX.DeclareKernelEntry kernel_path("flash_sdpa") -> !VPURegMapped.Index<0:0:0>
    %range = VPUMI40XX.ActKernelRange kernel_entry_index(%entry : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>

    %mi = VPUMI40XX.MappedInference dmas((%dma0) : (!VPURegMapped.Index<0:0:0>)) actKernelRanges((%range) : (!VPURegMapped.Index<0:0:0>)) barriers(%bar : !VPURegMapped.Index<0:0:0>) dmaCount([[2]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([[1]]) actKernelInvocationsCount([[0]]) mediaCount(0) barrierCount(1) -> !VPURegMapped.Index<0:0:0>

    return %arg1 : memref<1x16x16x16xf16, @DDR>
  }

  // The 928 bytes of DpuParamsBuffers are shared by all tiles, hence a single constant
  // CHECK:       [[CST:%.+]] = const.Declare memref<232xui32>

  // One buffer per tile, all at the dedicated 0x12600 offset
  // CHECK:       [[BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <75264> -> memref<232xui32, [@CMX_NN, 0]>

  // Bootstrap DMAs, chained ahead of the original first DMA. start_after immediately follows
  // outputs, which is what proves there are no wait/update barriers on them
  // CHECK:       [[PARAMS_DMA0:%.+]] = VPUMI40XX.NNDMA <{dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 928 : i64, srcWidth = 928 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 928 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64, wlmPage = -1 : i64}>
  // CHECK-SAME:      inputs([[CST]] : memref<232xui32>) outputs([[BUF0]] : memref<232xui32, [@CMX_NN, 0]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

  // MappedInference points at the new list head and accounts for the extra DMAs
  // CHECK:       VPUMI40XX.MappedInference dmas(([[PARAMS_DMA0]]) : (!VPURegMapped.Index<0:0:0>))
  // CHECK-SAME:      dmaCount({{\[\[}}3{{\]\]}})
}

// -----

// No SHAVE task uses the DPU, so the pass leaves the IR untouched

module @ShaveDpuDescriptorsWithoutDpuShaveTask attributes {config.compilationMode = #config.compilation_mode<DefaultHW>, config.revisionID = #config.revision_id<REVISION_NONE>} {
  config.Resources 2 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.ExecutorResource 1 of @M2I
  config.ExecutorResource 1 of @DMA_NN
  config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x16x16xf16>
  }
  func.func @main(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<1x16x16x16xf16, @DDR>) -> memref<1x16x16x16xf16, @DDR> {
    %in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x16x16xf16, @DDR>
    %out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x16x16xf16, @DDR>
    %cmx = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, [@CMX_NN, 0]>

    %bar = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}> <0, -1> -> !VPURegMapped.Index<0:0:0>

    %dma0 = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs(%in : memref<1x16x16x16xf16, @DDR>) outputs(%cmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) updates(%bar : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %dma1 = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs(%cmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) outputs(%out : memref<1x16x16x16xf16, @DDR>) previousDMA(%dma0 : !VPURegMapped.Index<0:0:0>) waits(%bar : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>

    %entry = VPUMI40XX.DeclareKernelEntry kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
    %range = VPUMI40XX.ActKernelRange kernel_entry_index(%entry : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>

    %mi = VPUMI40XX.MappedInference dmas((%dma0) : (!VPURegMapped.Index<0:0:0>)) actKernelRanges((%range) : (!VPURegMapped.Index<0:0:0>)) barriers(%bar : !VPURegMapped.Index<0:0:0>) dmaCount([[2]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([[1]]) actKernelInvocationsCount([[0]]) mediaCount(0) barrierCount(1) -> !VPURegMapped.Index<0:0:0>

    return %arg1 : memref<1x16x16x16xf16, @DDR>
  }

  // CHECK-NOT:   memref<232xui32>
  // CHECK-NOT:   <75264>
  // CHECK:       VPUMI40XX.MappedInference
  // CHECK-SAME:      dmaCount({{\[\[}}2{{\]\]}})
}

// -----

// Three tile configuration - one buffer and one DMA per tile, all sharing the same constant

module @ShaveDpuDescriptorsThreeTiles attributes {config.compilationMode = #config.compilation_mode<DefaultHW>, config.revisionID = #config.revision_id<REVISION_NONE>} {
  config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.ExecutorResource 1 of @M2I
  config.ExecutorResource 1 of @DMA_NN
  config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x16x16xf16>
  }
  func.func @main(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<1x16x16x16xf16, @DDR>) -> memref<1x16x16x16xf16, @DDR> {
    %in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x16x16xf16, @DDR>
    %out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x16x16xf16, @DDR>
    %cmx = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, [@CMX_NN, 0]>

    %bar = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}> <0, -1> -> !VPURegMapped.Index<0:0:0>

    %dma0 = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs(%in : memref<1x16x16x16xf16, @DDR>) outputs(%cmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) updates(%bar : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %dma1 = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs(%cmx : memref<1x16x16x16xf16, [@CMX_NN, 0]>) outputs(%out : memref<1x16x16x16xf16, @DDR>) previousDMA(%dma0 : !VPURegMapped.Index<0:0:0>) waits(%bar : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>

    %entry = VPUMI40XX.DeclareKernelEntry kernel_path("attention") -> !VPURegMapped.Index<0:0:0>
    %range = VPUMI40XX.ActKernelRange kernel_entry_index(%entry : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>

    %mi = VPUMI40XX.MappedInference dmas((%dma0) : (!VPURegMapped.Index<0:0:0>)) actKernelRanges((%range) : (!VPURegMapped.Index<0:0:0>)) barriers(%bar : !VPURegMapped.Index<0:0:0>) dmaCount([[2]]) invariantCount([0]) variantCount([0]) actKernelRangesCount([[1]]) actKernelInvocationsCount([[0]]) mediaCount(0) barrierCount(1) -> !VPURegMapped.Index<0:0:0>

    return %arg1 : memref<1x16x16x16xf16, @DDR>
  }

  // Still a single shared constant, regardless of the tile count
  // CHECK:       [[CST:%.+]] = const.Declare memref<232xui32>

  // CHECK:       [[BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <75264> -> memref<232xui32, [@CMX_NN, 0]>
  // CHECK:       [[BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <75264> -> memref<232xui32, [@CMX_NN, 1]>
  // CHECK:       [[BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <75264> -> memref<232xui32, [@CMX_NN, 2]>

  // CHECK:       [[PARAMS_DMA0:%.+]] = VPUMI40XX.NNDMA
  // CHECK-SAME:      wlmPage = -1 : i64
  // CHECK-SAME:      inputs([[CST]] : memref<232xui32>) outputs([[BUF0]] : memref<232xui32, [@CMX_NN, 0]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

  // CHECK:       [[PARAMS_DMA1:%.+]] = VPUMI40XX.NNDMA
  // CHECK-SAME:      inputs([[CST]] : memref<232xui32>) outputs([[BUF1]] : memref<232xui32, [@CMX_NN, 1]>) previousDMA([[PARAMS_DMA0]] : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>

  // CHECK:       [[PARAMS_DMA2:%.+]] = VPUMI40XX.NNDMA
  // CHECK-SAME:      inputs([[CST]] : memref<232xui32>) outputs([[BUF2]] : memref<232xui32, [@CMX_NN, 2]>) previousDMA([[PARAMS_DMA1]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>

  // The two original DMAs follow the three new ones
  // CHECK:       [[ORIG_DMA0:%.+]] = VPUMI40XX.NNDMA
  // CHECK-SAME:      previousDMA([[PARAMS_DMA2]] : !VPURegMapped.Index<0:0:2>)
  // CHECK-SAME:      -> !VPURegMapped.Index<0:0:3>
  // CHECK:       VPUMI40XX.NNDMA
  // CHECK-SAME:      previousDMA([[ORIG_DMA0]] : !VPURegMapped.Index<0:0:3>)
  // CHECK-SAME:      -> !VPURegMapped.Index<0:0:4>

  // CHECK:       VPUMI40XX.MappedInference dmas(([[PARAMS_DMA0]]) : (!VPURegMapped.Index<0:0:0>))
  // CHECK-SAME:      dmaCount({{\[\[}}5{{\]\]}})
}
