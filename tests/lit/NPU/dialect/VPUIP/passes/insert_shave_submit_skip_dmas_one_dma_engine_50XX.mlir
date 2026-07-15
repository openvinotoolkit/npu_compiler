//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --platform=%platform% --insert-shave-submit-skip-dmas="num-dma-engines=1" %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @Activation attributes {config.compilationMode = #config.compilation_mode<DefaultHW>, config.elf_version = #config.version<2 : 0 : 0>, config.platform = #config.platform<NPU5010>, config.revisionID = #config.revision_id<REVISION_NONE>} {
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_AtanDma(memref<*xf16>, memref<*xsi32>, memref<*xui8, [@CMX_NN, 0]>, memref<*xf16>, memref<*xsi32>, memref<*xui8, [@CMX_NN, 0]>, i64, i64, i64) attributes {VPU.kernel_code = "activation_atan_dma.cpp", VPU.kernel_entry = "activation_atan_dma", VPU.kernel_name = "activation_atan_dma", VPU.task_type = @COMPUTE}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }
  config.PipelineOptions @Options {
    config.Option @config.WorkloadManagementStatus : "ENABLED"
    config.Option @config.UseDedicatedFifoPerShaveEngine : true
  }
  config.Resources 1 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.Resources 1 of @global {
    config.ExecutorResource 1 of @M2I
    config.ExecutorResource 1 of @DMA_NN
    config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  }
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Input" tensorNames = ["dyn_input"] : tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 4194304]> : tensor<4xsi64>, order = #NCHW}>
    DataInfo "vpux_ie_shape_Input" : tensor<4xsi32>
  } outputsInfo : {
    DataInfo "bar1" friendlyName = "Result_12" : tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 4194304]> : tensor<4xsi64>, order = #NCHW}>
    DataInfo "vpux_ie_shape_bar1" : tensor<4xsi32>
  }
  func.func @main(%main: memref<1x1x1x4194304xf16, @DDR>, %main_0: memref<4xsi32, @DDR>, %main_1: memref<1x1x1x4194304xf16, @DDR>, %main_2: memref<4xsi32, @DDR>) -> (memref<1x1x1x4194304xf16, @DDR>, memref<4xsi32, @DDR>) {
    %bar0 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x1x2097152xf16, {order = #NCHW, strides = [4194304, 4194304, 4194304, 1]}, @DDR>
    %1 = VPURT.DeclareBuffer <NetworkInput> [0] <4194304> -> memref<1x1x1x2097152xf16, {order = #NCHW, strides = [4194304, 4194304, 4194304, 1]}, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkInput> [1] <0> -> memref<4xsi32, @DDR>
    %3 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1x1x2097152xf16, {order = #NCHW, strides = [4194304, 4194304, 4194304, 1]}, @DDR>
    %4 = VPURT.DeclareBuffer <NetworkOutput> [0] <4194304> -> memref<1x1x1x2097152xf16, {order = #NCHW, strides = [4194304, 4194304, 4194304, 1]}, @DDR>
    %5 = VPURT.DeclareBuffer <NetworkOutput> [1] <0> -> memref<4xsi32, @DDR>
    %6 = VPURT.DeclareBuffer <DDR> <8388672> -> memref<1x1x1x2097152xf16, {order = #NCHW, strides = [4194304, 4194304, 4194304, 1]}, @DDR>
    %7 = VPURT.DeclareBuffer <DDR> <12582976> -> memref<1x1x1x2097152xf16, {order = #NCHW, strides = [4194304, 4194304, 4194304, 1]}, @DDR>
    %8 = VPURT.DeclareBuffer <DDR> <8388672> -> memref<1x1x1x4194304xf16, @DDR>
    %9 = VPURT.DeclareBuffer <DDR> <16777280> -> memref<4xsi32, @DDR>
    %10 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %11 = VPURT.DeclareBuffer <DDR> <64> -> memref<1x1x1x2097152xf16, {order = #NCHW, strides = [4194304, 4194304, 4194304, 1]}, @DDR>
    %12 = VPURT.DeclareBuffer <DDR> <4194368> -> memref<1x1x1x2097152xf16, {order = #NCHW, strides = [4194304, 4194304, 4194304, 1]}, @DDR>
    %13 = VPURT.DeclareBuffer <DDR> <64> -> memref<1x1x1x4194304xf16, @DDR>
    %14 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1049600xui8, [@CMX_NN, 0]>
    %16 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<0x0x0x0xi32, [@CMX_NN, 0]>
    %17 = VPURT.DeclareBuffer <DDR> <0> -> memref<4xsi32, @DDR>
    VPURT.Task  {
      %15 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%2 : memref<4xsi32, @DDR>) outputs(%17 : memref<4xsi32, @DDR>) -> memref<4xsi32, @DDR>
    }
    VPURT.Task updates(%bar0 : !VPURT.Barrier)  {
      %15 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%0 : memref<1x1x1x2097152xf16, {order = #NCHW, strides = [4194304, 4194304, 4194304, 1]}, @DDR>) outputs(%11 : memref<1x1x1x2097152xf16, {order = #NCHW, strides = [4194304, 4194304, 4194304, 1]}, @DDR>) -> memref<1x1x1x2097152xf16, {order = #NCHW, strides = [4194304, 4194304, 4194304, 1]}, @DDR>
    }
    VPURT.Task updates(%bar0 : !VPURT.Barrier)  {
      %15 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%1 : memref<1x1x1x2097152xf16, {order = #NCHW, strides = [4194304, 4194304, 4194304, 1]}, @DDR>) outputs(%12 : memref<1x1x1x2097152xf16, {order = #NCHW, strides = [4194304, 4194304, 4194304, 1]}, @DDR>) -> memref<1x1x1x2097152xf16, {order = #NCHW, strides = [4194304, 4194304, 4194304, 1]}, @DDR>
    }
    VPURT.Task updates(%bar2 : !VPURT.Barrier)  {
      %15 = VPUIP.FetchDMA <{port = 0 : i64}> inputs(%10 : memref<0x0x0x0xi32, @DDR>) outputs(%10 : memref<0x0x0x0xi32, @DDR>) fetch_dma(<<DMA_NN>, tile = 0 : i64, list = 0 : i64, fetchType = <SingleDescriptor>, logicalTaskIdx = 0 : i64, descId = 0 : i64>) -> memref<0x0x0x0xi32, @DDR>
    }
    VPURT.Task updates(%bar2 : !VPURT.Barrier)  {
      %15 = VPUIP.FetchDMA <{port = 0 : i64}> inputs(%10 : memref<0x0x0x0xi32, @DDR>) outputs(%10 : memref<0x0x0x0xi32, @DDR>) fetch_dma(<<DMA_NN>, tile = 0 : i64, list = 0 : i64, fetchType = <SingleDescriptor>, logicalTaskIdx = 0 : i64, descId = 1 : i64>) -> memref<0x0x0x0xi32, @DDR>
    }
    VPURT.Task updates(%bar2 : !VPURT.Barrier)  {
      %15 = VPUIP.FetchDMA <{port = 0 : i64}> inputs(%10 : memref<0x0x0x0xi32, @DDR>) outputs(%10 : memref<0x0x0x0xi32, @DDR>) fetch_dma(<<DMA_NN>, tile = 0 : i64, list = 1 : i64, fetchType = <SingleDescriptor>, logicalTaskIdx = 0 : i64, descId = 2 : i64>) -> memref<0x0x0x0xi32, @DDR>
    }
    VPURT.Task updates(%bar2 : !VPURT.Barrier)  {
      %15 = VPUIP.FetchDMA <{port = 0 : i64}> inputs(%10 : memref<0x0x0x0xi32, @DDR>) outputs(%10 : memref<0x0x0x0xi32, @DDR>) fetch_dma(<<DMA_NN>, tile = 0 : i64, list = 1 : i64, fetchType = <SingleDescriptor>, logicalTaskIdx = 0 : i64, descId = 3 : i64>) -> memref<0x0x0x0xi32, @DDR>
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
      %15 = VPUIP.SyncDMA {logical_task = 0 : i64} <{port = 0 : i64}> inputs(%10 : memref<0x0x0x0xi32, @DDR>) outputs(%10 : memref<0x0x0x0xi32, @DDR>) -> memref<0x0x0x0xi32, @DDR>
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
      %15 = VPUIP.SyncDMA {logical_task = 0 : i64} <{port = 0 : i64}> inputs(%16 : memref<0x0x0x0xi32, [@CMX_NN, 0]>) outputs(%10 : memref<0x0x0x0xi32, @DDR>) -> memref<0x0x0x0xi32, @DDR>
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier)  {
      %bar1_19:2, %bar1_20 = VPUIP.SW.Kernel {dynamicInputShapesMap = array<i32: 0, -1>, dynamicOutputShapesMap = array<i32: 0, -1>, logical_task = 0 : i64, resultSegmentSizes = array<i32: 2, 1, 0>, skipDescIds = [0, 1]} @VPU.SW::@builtin_AtanDma inputs(%13 as %main_21: memref<1x1x1x4194304xf16, @DDR>, %14 as %bar1_22: memref<1x1x1x1049600xui8, [@CMX_NN, 0]>) dynamicInputShapes(%9 : memref<4xsi32, @DDR>) outputs(%8 as %main_23: memref<1x1x1x4194304xf16, @DDR>, %14 as %bar1_24: memref<1x1x1x1049600xui8, [@CMX_NN, 0]>) dynamicOutputShapes(%9 : memref<4xsi32, @DDR>) on tile 0 list 0 -> (memref<1x1x1x4194304xf16, @DDR>, memref<1x1x1x1049600xui8, [@CMX_NN, 0]>, memref<4xsi32, @DDR>){
        VPUIP.SW.Kernel.run {attrs = [8589934593, 999, 999, 8589934593, 999, 999]}(%main_21, %bar1_22, %main_23, %bar1_24) : memref<1x1x1x4194304xf16, @DDR>, memref<1x1x1x1049600xui8, [@CMX_NN, 0]>, memref<1x1x1x4194304xf16, @DDR>, memref<1x1x1x1049600xui8, [@CMX_NN, 0]>
      }
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier)  {
      %bar1_19:2, %bar1_20 = VPUIP.SW.Kernel {dynamicInputShapesMap = array<i32: 0, -1>, dynamicOutputShapesMap = array<i32: 0, -1>, logical_task = 0 : i64, resultSegmentSizes = array<i32: 2, 1, 0>, skipDescIds = [2, 3]} @VPU.SW::@builtin_AtanDma inputs(%13 as %main_21: memref<1x1x1x4194304xf16, @DDR>, %14 as %bar1_22: memref<1x1x1x1049600xui8, [@CMX_NN, 0]>) dynamicInputShapes(%9 : memref<4xsi32, @DDR>) outputs(%8 as %main_23: memref<1x1x1x4194304xf16, @DDR>, %14 as %bar1_24: memref<1x1x1x1049600xui8, [@CMX_NN, 0]>) dynamicOutputShapes(%9 : memref<4xsi32, @DDR>) on tile 0 list 1 -> (memref<1x1x1x4194304xf16, @DDR>, memref<1x1x1x1049600xui8, [@CMX_NN, 0]>, memref<4xsi32, @DDR>){
        VPUIP.SW.Kernel.run {attrs = [8589934593, 999, 999, 8589934593, 999, 999]}(%main_21, %bar1_22, %main_23, %bar1_24) : memref<1x1x1x4194304xf16, @DDR>, memref<1x1x1x1049600xui8, [@CMX_NN, 0]>, memref<1x1x1x4194304xf16, @DDR>, memref<1x1x1x1049600xui8, [@CMX_NN, 0]>
      }
    }
    VPURT.Task waits(%bar1 : !VPURT.Barrier)  {
      %15 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%9 : memref<4xsi32, @DDR>) outputs(%5 : memref<4xsi32, @DDR>) -> memref<4xsi32, @DDR>
    }
    VPURT.Task waits(%bar1 : !VPURT.Barrier)  {
      %15 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%6 : memref<1x1x1x2097152xf16, {order = #NCHW, strides = [4194304, 4194304, 4194304, 1]}, @DDR>) outputs(%3 : memref<1x1x1x2097152xf16, {order = #NCHW, strides = [4194304, 4194304, 4194304, 1]}, @DDR>) -> memref<1x1x1x2097152xf16, {order = #NCHW, strides = [4194304, 4194304, 4194304, 1]}, @DDR>
    }
    VPURT.Task  {
      %15 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%7 : memref<1x1x1x2097152xf16, {order = #NCHW, strides = [4194304, 4194304, 4194304, 1]}, @DDR>) outputs(%4 : memref<1x1x1x2097152xf16, {order = #NCHW, strides = [4194304, 4194304, 4194304, 1]}, @DDR>) -> memref<1x1x1x2097152xf16, {order = #NCHW, strides = [4194304, 4194304, 4194304, 1]}, @DDR>
    }
    return %main_1, %main_2 : memref<1x1x1x4194304xf16, @DDR>, memref<4xsi32, @DDR>
  }

  // CHECK: [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier
  // CHECK: [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier
  // CHECK: [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier

  // DDR Sync
  // CHECK: VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier)
  // CHECK: VPUIP.SyncDMA {logical_task = 0 : i64} <{port = 0 : i64}> inputs(%16 : memref<0x0x0x0xi32, @DDR>) outputs(%16 : memref<0x0x0x0xi32, @DDR>) -> memref<0x0x0x0xi32, @DDR>
  // CHECK-DAG: VPUIP.SkipDMA <{port = 0 : i64}> inputs(%4 : memref<0x0x0x0xi32, @DDR>) outputs(%3 : memref<0x0x0x0xi32, @DDR>) skip_dma(<tile = 0 : i64, list = 1 : i64, logicalTask = 0 : i64, descId = 2 : i64>) -> memref<0x0x0x0xi32, @DDR>
  // CHECK-DAG: VPUIP.SkipDMA <{port = 0 : i64}> inputs(%4 : memref<0x0x0x0xi32, @DDR>) outputs(%3 : memref<0x0x0x0xi32, @DDR>) skip_dma(<tile = 0 : i64, list = 0 : i64, logicalTask = 0 : i64, descId = 0 : i64>) -> memref<0x0x0x0xi32, @DDR>

  // CMX Sync
  // CHECK: VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier)
  // VPUIP.SyncDMA {logical_task = 0 : i64} <{port = 0 : i64}> inputs(%21 : memref<0x0x0x0xi32, [@CMX_NN, 0]>) outputs(%16 : memref<0x0x0x0xi32, @DDR>) -> memref<0x0x0x0xi32, @DDR>
  // CHECK-DAG: VPUIP.SkipDMA <{port = 0 : i64}> inputs(%5 : memref<0x0x0x0xi32, [@CMX_NN, 0]>) outputs(%3 : memref<0x0x0x0xi32, @DDR>) skip_dma(<tile = 0 : i64, list = 1 : i64, logicalTask = 0 : i64, descId = 3 : i64>) -> memref<0x0x0x0xi32, @DDR>
  // CHECK-DAG: VPUIP.SkipDMA <{port = 0 : i64}> inputs(%5 : memref<0x0x0x0xi32, [@CMX_NN, 0]>) outputs(%3 : memref<0x0x0x0xi32, @DDR>) skip_dma(<tile = 0 : i64, list = 0 : i64, logicalTask = 0 : i64, descId = 1 : i64>) -> memref<0x0x0x0xi32, @DDR>
}
