//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --wlm-split-graph-to-pages="num-barriers=4" %s | FileCheck %s
// REQUIRES: platform-NPU4000

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
func.func nested @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>) attributes {
     VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE
     }
}

func.func @DmaAndShvGraph() -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier <{isStartBarrier}> -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar6 = VPURT.DeclareVirtualBarrier <{isFinalBarrier}> -> !VPURT.Barrier

    // dummy buffer
    %cst0 = const.Declare memref<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf1 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf2 = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %buf3 = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // Simple subgraph with dummy ops:
    //             DMA0
    //              |
    //             bar0
    //              |
    //             DMA1
    //              |
    //             bar1
    //             /  \   \
    //          DMA2  SHV0 \
    //           |     |    \
    //          bar2  bar3  |
    //           |     |    |
    //          SHV1  DMA3 SHV2   <- Since SHV2 wait on bar1 from Page0, original page assignment
    //            \    /  /          will be Page0, but WlmSplitGraphToPages pass should reassign it
    //             bar4              to Page1 so that it is not smaller than some previous SHV task
    //              |                on same tile - SHV1 executes on Page1.
    //             SHV3
    //              |
    //             bar5
    //              |
    //             DMA4
    //              |
    //             bar6

    VPURT.Task updates(%bar0: !VPURT.Barrier)
    {
         VPUIP.NNDMA inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier)
    {
         VPUIP.NNDMA inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier)
    {
         VPUIP.NNDMA inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier)
    {
          VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
              inputs(%buf0 as %input: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
              outputs(%buf0 as %output: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]> {
              VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%input, %output) : memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
          }
    }

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier)
    {
          VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
              inputs(%buf0 as %input: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
              outputs(%buf0 as %output: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]> {
              VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%input, %output) : memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
          }
    }

    VPURT.Task waits(%bar3: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier)
    {
         VPUIP.NNDMA inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier)
    {
          VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
              inputs(%buf0 as %input: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
              outputs(%buf0 as %output: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]> {
              VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%input, %output) : memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
          }
    }

    VPURT.Task waits(%bar4: !VPURT.Barrier) updates(%bar5: !VPURT.Barrier)
    {
          VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
              inputs(%buf0 as %input: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
              outputs(%buf0 as %output: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]> {
              VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%input, %output) : memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
          }
    }

    VPURT.Task waits(%bar5: !VPURT.Barrier) updates(%bar6: !VPURT.Barrier)
    {
         VPUIP.NNDMA inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }


    return %buf3: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // After page split (barX(PageY)):
    //
    // _______      DMA0
    //               |
    //              bar0(0)
    //               |
    //  Page0       DMA1
    //               |
    //              bar1(0)
    //              /     \
    // _______     DMA2   SHV0
    //             |        |
    //             |        |
    //  Page1    bar2(1)   bar3(1)
    //            /  \      |
    // _______   SHV1 SHV2  DMA3
    //             \  |    /
    //              bar4(2)
    //               |
    //  Page2       SHV3
    //               |
    //              bar5(2)
    //               |
    // _______      DMA4
    //               |
    //  Page3       bar6(3)
    // _______

    // CHECK:   [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier <{isStartBarrier, wlmPage = 0 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR4:%.+]] = VPURT.DeclareVirtualBarrier <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR5:%.+]] = VPURT.DeclareVirtualBarrier <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR6:%.+]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier, wlmPage = 3 : i64}> -> !VPURT.Barrier

    // CHECK:   VPURT.Task updates([[BAR0]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT:     VPUIP.NNDMA
    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT:     VPUIP.NNDMA
    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT:     VPUIP.NNDMA
    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT:     VPUIP.SW.Kernel
    // CHECK:   VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT:     VPUIP.NNDMA
    // CHECK:   VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT:     VPUIP.SW.Kernel
    // CHECK:   VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT:     VPUIP.SW.Kernel
    // CHECK:   VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK-NEXT:     VPUIP.SW.Kernel
    // CHECK:   VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK-NEXT:     VPUIP.NNDMA
}
