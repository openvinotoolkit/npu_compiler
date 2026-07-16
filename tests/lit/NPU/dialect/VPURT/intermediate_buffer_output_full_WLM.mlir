//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform% allow-custom-values=true" --intermediate-buffer-output="op-index=3 insertion-index=3 buffer-index=1" %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType2 = !quant.uniform<u8:f16, 0.01269696927538105>
!qElemType3 = !quant.uniform<u8:f16, 0.0048000719033035573>
!qElemType4 = !quant.uniform<u8:f16, 0.0173492431640625:114>
!qElemType5 = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType6 = !quant.uniform<u8:f16, 0.0024000359516517787>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

module @TestBufferOutput {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x3x224x224xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x224x224xf16>
    // CHECK: DataInfo "output" : tensor<1x3x224x224xf16>
  }

  module @VPU.SW {
  func.func nested @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>) attributes {
          VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE
      }
  }

  func.func @main(%arg0: memref<1x3x224x224xf16, @DDR>, %arg1: memref<1x3x224x224xf16, @DDR>) -> memref<1x3x224x224xf16, @DDR> {
  // CHECK: func.func @main([[ARG0:%.+]]: memref<1x3x224x224xf16, @DDR>, [[ARG1:%.+]]: memref<1x3x224x224xf16, @DDR>) -> memref<1x3x224x224xf16, @DDR> {

    %bar0 = VPURT.ConfigureBarrier<0> <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    %bar1 = VPURT.ConfigureBarrier<1> <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    %bar2 = VPURT.ConfigureBarrier<2> <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    %bar3 = VPURT.ConfigureBarrier<3> <{isFinalBarrier, wlmPage = 0 : i64}> -> !VPURT.Barrier

    %net_input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x224x224xf16, @DDR>
    %net_output = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x224x224xf16, @DDR>

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x224x224xf16, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <301056> -> memref<1x3x224x224xf16, @DDR>
    %buf2 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x224x224xf16, @DDR>

    %dummy_buf = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>

    VPURT.Task wlmPage(0)  {
      %0 = VPUIP.BarProgDMA <{port = 0 : i64}> inputs(%dummy_buf : memref<0x0x0x0xi32, @DDR>) outputs(%dummy_buf : memref<0x0x0x0xi32, @DDR>) physical_barrier_range(<0 : i64 to 47 : i64>) -> memref<0x0x0x0xi32, @DDR>
    }
    VPURT.Task updates(%bar0 : !VPURT.Barrier) wlmPage(0)  {
      %0 = VPUIP.FetchDMA <{port = 0 : i64}> inputs(%dummy_buf : memref<0x0x0x0xi32, @DDR>) outputs(%dummy_buf : memref<0x0x0x0xi32, @DDR>) fetch_dma(<<SHAVE_ACT>, tile = 1 : i64, list = 0 : i64, fetchType = <DescriptorGroup>, group = 0 : i64>) -> memref<0x0x0x0xi32, @DDR>
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) wlmPage(0)  {
      %0 = VPUIP.EnqueueDMA <{port = 0 : i64}> inputs(%dummy_buf : memref<0x0x0x0xi32, @DDR>) outputs(%dummy_buf : memref<0x0x0x0xi32, @DDR>) enqueue_dma_attr(<<SHAVE_ACT>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 138 : i64>) -> memref<0x0x0x0xi32, @DDR>
    }
    VPURT.Task updates(%bar1 : !VPURT.Barrier) wlmPage(0)  {
      %0 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%net_input : memref<1x3x224x224xf16, @DDR>) outputs(%buf0 : memref<1x3x224x224xf16, @DDR>) -> memref<1x3x224x224xf16, @DDR>
    }
    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) wlmPage(0)  {
      %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
          inputs(%buf0 as %input: memref<1x3x224x224xf16, @DDR>)
          outputs(%buf1 as %output: memref<1x3x224x224xf16, @DDR>) on tile 0 list 0 -> memref<1x3x224x224xf16, @DDR> {
          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%input, %output) : memref<1x3x224x224xf16, @DDR>, memref<1x3x224x224xf16, @DDR>
      }
    }
    VPURT.Task waits(%bar2 : !VPURT.Barrier) updates(%bar3 : !VPURT.Barrier) wlmPage(0)  {
      %0 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%buf2 : memref<1x3x224x224xf16, @DDR>) outputs(%net_output : memref<1x3x224x224xf16, @DDR>) -> memref<1x3x224x224xf16, @DDR>
    }
    return %arg1 : memref<1x3x224x224xf16, @DDR>
  }

  // CHECK:       [[BAR0:%.+]] = VPURT.ConfigureBarrier<0>
  // CHECK:       [[BAR1:%.+]] = VPURT.ConfigureBarrier<1>
  // CHECK:       [[BAR2:%.+]] = VPURT.ConfigureBarrier<2>
  // CHECK-SAME:                  isFinalBarrier
  // CHECK-NOT:                  VPURT.ConfigureBarrier


  // CHECK:        VPURT.Task wlmPage(0)
  // CHECK-NEXT:     VPUIP.BarProgDMA

  // Fetch DMA converted to SyncDMA
  // CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) wlmPage(0)
  // CHECK-NEXT:     VPUIP.SyncDMA

  // Enqueue DMA converted to SyncDMA
  // CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) wlmPage(0)
  // CHECK-NEXT:     VPUIP.SyncDMA

  // Target TaskOp
  // CHECK:       VPURT.Task updates([[BAR1]] : !VPURT.Barrier)  wlmPage(0) <{taskIndex = [3]}>
  // CHECK-NEXT:     VPUIP.NNDMA

  // Inserted copy out DMA
  // CHECK:       VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
  // CHECK:          VPUIP.NNDMA

  // CHECK-NOT:   VPURT.Task

  // Updated return
  // CHECK:   return [[ARG1]] : memref<1x3x224x224xf16, @DDR>
}
