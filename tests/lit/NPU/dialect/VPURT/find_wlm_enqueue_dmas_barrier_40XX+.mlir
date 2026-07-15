//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --find-wlm-enqueue-dmas-barrier %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DmaAndDpuGraph() -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]> {
    %bar0 = VPURT.ConfigureBarrier<0> <{isStartBarrier, wlmPage = 0 : i64}> -> !VPURT.Barrier
    %bar1 = VPURT.ConfigureBarrier<1> <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    %bar2 = VPURT.ConfigureBarrier<2> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    %bar3 = VPURT.ConfigureBarrier<3> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    %bar4 = VPURT.ConfigureBarrier<0> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    %bar5 = VPURT.ConfigureBarrier<1> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    %bar6 = VPURT.ConfigureBarrier<2> <{wlmPage = 3 : i64}> -> !VPURT.Barrier
    %bar7 = VPURT.ConfigureBarrier<3> <{isFinalBarrier, wlmPage = 3 : i64}> -> !VPURT.Barrier

    // dummy buffer
    %cst0 = const.Declare memref<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf1 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf2 = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %buf3 = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // Simple subgraph with dummy ops:
    //  DMA/DPU<HwFifo>[index]
    //  bar<index>(PID)
    //
    // _______      DMA0[0]
    //               |
    //              bar0(0)
    //               |
    //  Page0       DMA1[0]
    //               |
    //              bar1(1)
    //               |
    // _______      DMA0[1]
    //               |
    //              bar2(2)
    //               |
    //  Page1       DMA1[1]
    //               |
    //              bar3(3)
    //               |
    // _______      DMA0[2]
    //               |
    //              bar4(0)
    //               |
    //  Page2       DMA1[2]
    //               |
    //              bar5(1)
    //             /      \
    // _______  DMA0[3]   DPU0[0]
    //             \      /
    //              bar6(2)
    //               |
    //  Page3       DPU0[1]
    //               |
    //              bar7(3)
    // _______


    VPURT.Task updates(%bar0: !VPURT.Barrier) wlmPage(0)
    {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) wlmPage(0)
    {
        VPUIP.NNDMA <{port = 1 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) wlmPage(0)
    {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) wlmPage(1)
    {
        VPUIP.NNDMA <{port = 1 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar3: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) wlmPage(1)
    {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar4: !VPURT.Barrier) updates(%bar5: !VPURT.Barrier) wlmPage(2)
    {
        VPUIP.NNDMA <{port = 1 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar5: !VPURT.Barrier) updates(%bar6: !VPURT.Barrier) wlmPage(2)
    {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar5: !VPURT.Barrier) updates(%bar6: !VPURT.Barrier) wlmPage(2)
    {
        VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
            }>
            input(%buf0: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%buf2: memref<16x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%buf0: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%buf3: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%buf3: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants : {DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}
                        DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}} PPE : {}

    }

    VPURT.Task waits(%bar6: !VPURT.Barrier) updates(%bar7: !VPURT.Barrier) wlmPage(3)
    {
        VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
            }>
            input(%buf0: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%buf2: memref<16x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%buf0: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%buf3: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%buf3: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants : {DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}
                        DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}} PPE : {}

    }

    return %buf3: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // CHECK:   [[BAR0:%.+]] = VPURT.ConfigureBarrier<0> <{isStartBarrier, wlmPage = 0 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR1:%.+]] = VPURT.ConfigureBarrier<1> <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR2:%.+]] = VPURT.ConfigureBarrier<2> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR3:%.+]] = VPURT.ConfigureBarrier<3> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR4:%.+]] = VPURT.ConfigureBarrier<0> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR5:%.+]] = VPURT.ConfigureBarrier<1> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR6:%.+]] = VPURT.ConfigureBarrier<2> <{wlmPage = 3 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR7:%.+]] = VPURT.ConfigureBarrier<3> <{isFinalBarrier, wlmPage = 3 : i64}> -> !VPURT.Barrier


    // CHECK:   VPURT.Task updates([[BAR0]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 1 : i64}>
    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK:   VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 1 : i64}>
    // CHECK:   VPURT.Task waits([[BAR3]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.EnqueueDMA <{port = 0 : i64}>
    // CHECK-SAME: enqueue_dma_attr(<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 3 : i64>)
    // CHECK:   VPURT.Task updates([[BAR4]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK:   VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 1 : i64}>
    // CHECK:   VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK:   VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK-NEXT: VPUIP.NCEClusterTask
    // CHECK:   VPURT.Task waits([[BAR6]] : !VPURT.Barrier) updates([[BAR7]] : !VPURT.Barrier) wlmPage(3) {
    // CHECK-NEXT: VPUIP.NCEClusterTask

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DmaOnlyGraph() -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]> {
    %bar0 = VPURT.ConfigureBarrier<0> <{isStartBarrier, wlmPage = 0 : i64}> -> !VPURT.Barrier
    %bar1 = VPURT.ConfigureBarrier<1> <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    %bar2 = VPURT.ConfigureBarrier<2> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    %bar3 = VPURT.ConfigureBarrier<3> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    %bar4 = VPURT.ConfigureBarrier<0> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    %bar5 = VPURT.ConfigureBarrier<1> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    %bar6 = VPURT.ConfigureBarrier<2> <{isFinalBarrier, wlmPage = 3 : i64}> -> !VPURT.Barrier

    // dummy buffer
    %cst0 = const.Declare memref<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf1 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf2 = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %buf3 = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // Simple subgraph with dummy ops:
    //  DMA<HwFifo>[index]
    //  bar<index>(PID)
    //
    // _______      DMA0[0]
    //               |
    //              bar0(0)
    //               |
    //  Page0       DMA1[0]
    //               |
    //              bar1(1)
    //               |
    // _______      DMA0[1]
    //               |
    //              bar2(2)
    //               |
    //  Page1       DMA1[1]
    //               |
    //              bar3(3)
    //               |
    // _______      DMA0[2]
    //               |
    //              bar4(0)
    //               |
    //  Page2       DMA1[2]
    //               |
    //              bar5(1)
    //               |
    // _______      DMA0[3]
    //               |
    //  Page3       bar6(2)
    // _______


    VPURT.Task updates(%bar0: !VPURT.Barrier) wlmPage(0)
    {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) wlmPage(0)
    {
        VPUIP.NNDMA <{port = 1 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) wlmPage(0)
    {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) wlmPage(1)
    {
        VPUIP.NNDMA <{port = 1 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar3: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) wlmPage(1)
    {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar4: !VPURT.Barrier) updates(%bar5: !VPURT.Barrier) wlmPage(2)
    {
        VPUIP.NNDMA <{port = 1 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar5: !VPURT.Barrier) updates(%bar6: !VPURT.Barrier) wlmPage(2)
    {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    return %buf3: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // CHECK:   [[BAR0:%.+]] = VPURT.ConfigureBarrier<0> <{isStartBarrier, wlmPage = 0 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR1:%.+]] = VPURT.ConfigureBarrier<1> <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR2:%.+]] = VPURT.ConfigureBarrier<2> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR3:%.+]] = VPURT.ConfigureBarrier<3> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR4:%.+]] = VPURT.ConfigureBarrier<0> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR5:%.+]] = VPURT.ConfigureBarrier<1> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR6:%.+]] = VPURT.ConfigureBarrier<2> <{isFinalBarrier, wlmPage = 3 : i64}> -> !VPURT.Barrier

    // CHECK:   VPURT.Task updates([[BAR0]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 1 : i64}>
    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK:   VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 1 : i64}>
    // CHECK:   VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK:   VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 1 : i64}>
    // CHECK:   VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 0 : i64}>

}

// -----

// Simple subgraph:
    //  DMA<HwFifo>[index]
    //  bar<index>(PID)
    //
    // _______         DMA0[0]
    //                   |
    //                 bar0(0)
    //               /   |    \        ________
    //  Page0   DPU0[0]  |   DMA0[1]   Execution Group 0 (Has one task per queue)
    //              |    |     |       _________
    //              |  DPU0[1] |       Execution Group 1 (Has one task per queue)
    //              |    |     |       _________
    //               \   |    /
    //                 \ |   /
    //                 bar1(1)
    //                   |             _________
    //                 DPU0[2]         Execution Group 2 (Has one task per queue)
    // _______           |             _________
    //                   |
    //                 bar2(2)         _________
    //                 /   \           Execution Group 3 (Has one task per queue)
    //             DPU0[3] DMA0[2]     _________
    // Page 1          \   /
    //                 bar3(3)
    //                 /   \           _________
    //            DPU0[4] DPU1[4]
    // _______         \   /
    //                 bar4(0)         Execution Group 4
    //                 /   \
    //  Page2     DPU0[5] DMA0[3]
    //                 \   /           _________
    //                 bar5(1)
    //                   |
    // _______         DMA0[4]
    //                   |
    //                 bar6(2)
    // Page3             |
    //                 DMA0[5]
    //                   |
    //                 bar7(3)
    // _______

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 0.01269696927538105>
!qElemType2 = !quant.uniform<u8:f16, 0.0173492431640625:114>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @DpuEnqueueWithMultipleExecutionGroups attributes {} {
  config.PipelineOptions @Options {
    config.Option @config.MetadataMaxVariantCount : 8
    config.Option @config.MetadataMaxInvariantCount : 4
    config.Option @config.MetadataMaxKernelInvocationCount : 4
    config.Option @config.MetadataMaxKernelRangeCount : 4
  }
  func.func @main(%arg0: memref<1x3x224x224xf16, @DDR>, %arg1: memref<1x64x56x56xf16, @DDR>) -> memref<1x64x56x56xf16, @DDR> {
    %cst = const.Declare memref<1x1x1x5120xui8> = dense<1> : tensor<1x1x1x5120xui8>
    %0 = VPURT.ConfigureBarrier<0> <{isStartBarrier, wlmPage = 0 : i64}> -> !VPURT.Barrier
    %1 = VPURT.ConfigureBarrier<1> <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    %3 = VPURT.ConfigureBarrier<2> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    %4 = VPURT.ConfigureBarrier<3> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    %51 = VPURT.ConfigureBarrier<0> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    %52 = VPURT.ConfigureBarrier<1> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    %53 = VPURT.ConfigureBarrier<2> <{wlmPage = 3 : i64}> -> !VPURT.Barrier
    %5 = VPURT.ConfigureBarrier<3> <{isFinalBarrier, wlmPage = 3 : i64}> -> !VPURT.Barrier

    %6 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %7 = VPURT.DeclareBuffer <NetworkInput> [0] <48832> -> memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %8 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    %10 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x115x224xf16, [@CMX_NN, 1]>
    %11 = VPURT.DeclareBuffer <CMX_NN> <154560> -> !VPUIP.DistributedBuffer<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>
    %12 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x224x4x114x!qElemType, {order = #NWCH}, [@CMX_NN, 0]>
    %13 = VPURT.DeclareBuffer <CMX_NN> [0] <272960> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %14 = VPURT.DeclareBuffer <CMX_NN> <278528> {swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %15 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %16 = VPURT.DeclareBuffer <CMX_NN> [0] <278528> {swizzlingKey = 5 : i64} -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %17 = VPURT.DeclareBuffer <CMX_NN> <0> {swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x64x56x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %18 = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    %19 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <267840> -> !VPUIP.DistributedBuffer<1x1x1x5120xui8, {order = #NCHW, strides = [5120, 5120, 5120, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %20 = VPURT.DeclareBuffer <CMX_NN> [0] <200704> -> memref<1x64x28x56xf16, [@CMX_NN, 0]>
    %21 = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    %22 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %23 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %24 = VPURT.DeclareBuffer <CMX_NN> [0] <257600> -> memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %25 = VPURT.DeclareBuffer <CMX_NN> <154560> -> !VPUIP.DistributedBuffer<1x16x224x224x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    %26 = VPURT.DeclareBuffer <CMX_NN> [0] <154560> -> memref<1x16x114x224x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>
    VPURT.Task updates(%0 : !VPURT.Barrier) wlmPage(0) {
      %27 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%6 : memref<1x3x114x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%9 : memref<1x3x114x224xf16, [@CMX_NN, 0]>) -> memref<1x3x114x224xf16, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) wlmPage(0) {
      %27 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%7 : memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%10 : memref<1x3x115x224xf16, [@CMX_NN, 1]>) -> memref<1x3x115x224xf16, [@CMX_NN, 1]>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) wlmPage(0) {
      %27 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_permute_quantize, is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%23 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%22 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%21 : !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%11 : !VPUIP.DistributedBuffer<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>) outputs(%12 : memref<1x224x4x114x!qElemType, {order = #NWCH}, [@CMX_NN, 0]>) -> memref<1x224x4x114x!qElemType, {order = #NWCH}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) wlmPage(0) {
      %27 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_permute_quantize, is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%23 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%22 : memref<1x224x3x114xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%21 : !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%11 : !VPUIP.DistributedBuffer<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>) outputs(%12 : memref<1x224x4x114x!qElemType, {order = #NWCH}, [@CMX_NN, 0]>) -> memref<1x224x4x114x!qElemType, {order = #NWCH}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [113, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    VPURT.Task waits(%1: !VPURT.Barrier) updates(%3 : !VPURT.Barrier) wlmPage(0) {
      %27 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>}> input(%26 : memref<1x16x114x224x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>) weights(%24 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%25 : !VPUIP.DistributedBuffer<1x16x224x224x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, strides = [2, 2], num_clusters = 2 : i64}>) parent_output(%14 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%16 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) wlmPage(1) {
      %27 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%7 : memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%10 : memref<1x3x115x224xf16, [@CMX_NN, 1]>) -> memref<1x3x115x224xf16, [@CMX_NN, 1]>
    }
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) wlmPage(1) {
      %27 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_segmented, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>}> input(%15 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) parent_input(%14 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%17 : !VPUIP.DistributedBuffer<1x64x56x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%18 : memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%51 : !VPURT.Barrier) wlmPage(1) {
      %27 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_segmented, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>}> input(%15 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) parent_input(%14 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%17 : !VPUIP.DistributedBuffer<1x64x56x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%18 : memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    VPURT.Task waits(%51 : !VPURT.Barrier) updates(%52 : !VPURT.Barrier) wlmPage(2) {
      %27 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%7 : memref<1x3x115x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%10 : memref<1x3x115x224xf16, [@CMX_NN, 1]>) -> memref<1x3x115x224xf16, [@CMX_NN, 1]>
    }
    VPURT.Task waits(%51 : !VPURT.Barrier) updates(%52 : !VPURT.Barrier) wlmPage(2) {
      %27 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{is_segmented, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>}> input(%15 : memref<1x64x56x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) parent_input(%14 : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%17 : !VPUIP.DistributedBuffer<1x64x56x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%18 : memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>) -> memref<1x64x28x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%52 : !VPURT.Barrier) updates(%53 : !VPURT.Barrier) wlmPage(2) {
      %27 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%20 : memref<1x64x28x56xf16, [@CMX_NN, 0]>) outputs(%8 : memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>) -> memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>
    }
    VPURT.Task waits(%53 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) wlmPage(3) {
      %27 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%20 : memref<1x64x28x56xf16, [@CMX_NN, 0]>) outputs(%8 : memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>) -> memref<1x64x28x56xf16, {order = #NCHW, strides = [200704, 3136, 56, 1]}, @DDR>
    }
    return %arg1 : memref<1x64x56x56xf16, @DDR>
  }

  // CHECK: [[BAR0:%.+]] = VPURT.ConfigureBarrier<0> <{isStartBarrier, wlmPage = 0 : i64}> -> !VPURT.Barrier
  // CHECK: [[BAR1:%.+]] = VPURT.ConfigureBarrier<1> <{wlmPage = 0 : i64}> -> !VPURT.Barrier
  // CHECK: [[BAR2:%.+]] = VPURT.ConfigureBarrier<2> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
  // CHECK: [[BAR3:%.+]] = VPURT.ConfigureBarrier<3> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
  // CHECK: [[BAR4:%.+]] = VPURT.ConfigureBarrier<0> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
  // CHECK: [[BAR5:%.+]] = VPURT.ConfigureBarrier<1> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
  // CHECK: [[BAR6:%.+]] = VPURT.ConfigureBarrier<2> <{wlmPage = 3 : i64}> -> !VPURT.Barrier
  // CHECK: [[BAR7:%.+]] = VPURT.ConfigureBarrier<3> <{isFinalBarrier, wlmPage = 3 : i64}> -> !VPURT.Barrier

  // First DMA
  // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) wlmPage(0)
  // CHECK-NEXT: VPUIP.NNDMA

  // Enqueue DMAs for Group 0 and Group 1
  // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) wlmPage(0)
  // CHECK-NEXT: VPUIP.EnqueueDMA <{port = 0 : i64}>
  // CHECK-SAME: enqueue_dma_attr(<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 7 : i64>)

  // Group 0
  // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) wlmPage(0)
  // CHECK-NEXT: VPUIP.NCEClusterTask

  // Enqueue DMAs for Group 2
  // CHECK: VPURT.Task wlmPage(0)
  // CHECK-NEXT: VPUIP.EnqueueDMA <{port = 0 : i64}>
  // CHECK-SAME: enqueue_dma_attr(<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 8 : i64, endTask = 9 : i64>)

  // Group 1
  // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) wlmPage(0)
  // CHECK-NEXT: VPUIP.NCEClusterTask

  // Enqueue DMAs for Group 3
  // CHECK: VPURT.Task wlmPage(0)
  // CHECK-NEXT: VPUIP.EnqueueDMA <{port = 0 : i64}>
  // CHECK-SAME: enqueue_dma_attr(<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 10 : i64, endTask = 13 : i64>)

  // Enqueue DMAs for Group 4
  // CHECK: VPURT.Task wlmPage(0)
  // CHECK-NEXT: VPUIP.EnqueueDMA <{port = 0 : i64}>
  // CHECK-SAME: enqueue_dma_attr(<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 14 : i64, endTask = 17 : i64>)

  // CHECK: VPURT.Task wlmPage(0)
  // CHECK-NEXT: VPUIP.NNDMA

  // Group 2
  // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) wlmPage(0)
  // CHECK-NEXT: VPUIP.NCEClusterTask

  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) wlmPage(1)
  // CHECK-NEXT: VPUIP.NNDMA

  // Group 3
  // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) wlmPage(1)
  // CHECK-NEXT: VPUIP.NCEClusterTask

  // Group 4
  // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) wlmPage(1)
  // CHECK-NEXT: VPUIP.NCEClusterTask

  // CHECK: VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier) wlmPage(2)
  // CHECK-NEXT: VPUIP.NNDMA

  // CHECK: VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier) wlmPage(2)
  // CHECK-NEXT: VPUIP.NCEClusterTask

  // CHECK: VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier) wlmPage(2)
  // CHECK-NEXT: VPUIP.NNDMA

  // CHECK: VPURT.Task waits([[BAR6]] : !VPURT.Barrier) updates([[BAR7]] : !VPURT.Barrier) wlmPage(3)
  // CHECK-NEXT: VPUIP.NNDMA
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func nested @builtin_LstmDpu(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64) attributes {VPU.kernel_code = "lstm_dpu.cpp", VPU.kernel_entry = "lstm_dpu", VPU.task_type = @COMPUTE}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @ShvWithDpuGraph() -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]> {
    %bar0 = VPURT.ConfigureBarrier<0> <{isStartBarrier, wlmPage = 0 : i64}> -> !VPURT.Barrier
    %bar1 = VPURT.ConfigureBarrier<1> <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    %bar2 = VPURT.ConfigureBarrier<2> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    %bar3 = VPURT.ConfigureBarrier<3> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    %bar4 = VPURT.ConfigureBarrier<0> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    %bar5 = VPURT.ConfigureBarrier<1> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    %bar6 = VPURT.ConfigureBarrier<2> <{wlmPage = 3 : i64}> -> !VPURT.Barrier
    %bar7 = VPURT.ConfigureBarrier<3> <{isFinalBarrier, wlmPage = 3 : i64}> -> !VPURT.Barrier

    // dummy buffer
    %cst0 = const.Declare memref<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf1 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf2 = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %buf3 = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %buf4 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<1x1x2x256xf16, [@CMX_NN, 0]>
    %buf5 = VPURT.DeclareBuffer <CMX_NN> [0] <48448> -> memref<1x1x1x64xf16, [@CMX_NN, 0]>
    %buf6 = VPURT.DeclareBuffer <CMX_NN> [0] <48576> -> memref<1x1x1x64xf16, [@CMX_NN, 0]>
    %buf7 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x64x64xf16, [@CMX_NN, 0]>
    %buf8 = VPURT.DeclareBuffer <CMX_NN> [0] <48704> -> memref<1x1x1x2xsi32, [@CMX_NN, 0]>
    %buf9 = VPURT.DeclareBuffer <CMX_NN> [0] <40960> -> memref<1x1x1x1544xsi32, [@CMX_NN, 0]>
    %buf10 = VPURT.DeclareBuffer <CMX_NN> [0] <47168> -> memref<1x1x2x64xf16, [@CMX_NN, 0]>
    %buf11 = VPURT.DeclareBuffer <CMX_NN> [0] <47424> -> memref<1x1x1x64xf16, [@CMX_NN, 0]>
    %buf12 = VPURT.DeclareBuffer <CMX_NN> [0] <47552> -> memref<1x1x1x64xf16, [@CMX_NN, 0]>


    // Simple subgraph with dummy ops:
    //  DMA/DPU<HwFifo>[index]
    //  bar<index>(PID)
    //
    // _______      DMA0[0]
    //               |
    //              bar0(0)
    //               |
    //  Page0       DMA1[0]
    //               |
    //              bar1(1)
    //               |
    // _______      DMA0[1]
    //               |
    //              bar2(2)
    //               |
    //  Page1       DMA1[1]
    //               |
    //              bar3(3)
    //             /      \
    //       DMA0[2]     SHVwithDPU  <- DPU enqueue to be placed right after SHVwithDPU
    //              \     |             New EnqueueDMA will wait on bar4 and be assigned wlmPage 2
    //       DMA0[3] |    |          <- DMA0[3] wlmPage attribute to be changed from 1
    //  _______      |   /              to 2 after EnqueueDMA is inserted
    //              bar4(0)
    //               |
    //  Page2       DMA1[2]
    //               |
    //              bar5(1)
    //             /      \
    // _______  DMA0[4]   DPU0[0]
    //             \      /
    //              bar6(2)
    //               |
    //  Page3       DPU0[1]
    //               |
    //              bar7(3)
    // _______


    VPURT.Task updates(%bar0: !VPURT.Barrier) wlmPage(0)
    {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) wlmPage(0)
    {
        VPUIP.NNDMA <{port = 1 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) wlmPage(0)
    {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) wlmPage(1)
    {
        VPUIP.NNDMA <{port = 1 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar3: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) wlmPage(1)
    {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar3: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) wlmPage(1)
    {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 3, 0, 0>} @VPU.SW::@builtin_LstmDpu inputs(%buf4 as %arg6: memref<1x1x2x256xf16, [@CMX_NN, 0]>, %buf5 as %arg7: memref<1x1x1x64xf16, [@CMX_NN, 0]>, %buf6 as %arg8: memref<1x1x1x64xf16, [@CMX_NN, 0]>, %buf7 as %arg9: memref<1x4x64x64xf16, [@CMX_NN, 0]>, %buf8 as %arg10: memref<1x1x1x2xsi32, [@CMX_NN, 0]>, %buf9 as %arg11: memref<1x1x1x1544xsi32, [@CMX_NN, 0]>) outputs(%buf10 as %arg12: memref<1x1x2x64xf16, [@CMX_NN, 0]>, %buf11 as %arg13: memref<1x1x1x64xf16, [@CMX_NN, 0]>, %buf12 as %arg14: memref<1x1x1x64xf16, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x2x64xf16, [@CMX_NN, 0]>, memref<1x1x1x64xf16, [@CMX_NN, 0]>, memref<1x1x1x64xf16, [@CMX_NN, 0]>){
            VPUIP.SW.Kernel.run {attrs = [1, 52]}(%arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14) : memref<1x1x2x256xf16, [@CMX_NN, 0]>, memref<1x1x1x64xf16, [@CMX_NN, 0]>, memref<1x1x1x64xf16, [@CMX_NN, 0]>, memref<1x4x64x64xf16, [@CMX_NN, 0]>, memref<1x1x1x2xsi32, [@CMX_NN, 0]>, memref<1x1x1x1544xsi32, [@CMX_NN, 0]>, memref<1x1x2x64xf16, [@CMX_NN, 0]>, memref<1x1x1x64xf16, [@CMX_NN, 0]>, memref<1x1x1x64xf16, [@CMX_NN, 0]>
        }
    }

    VPURT.Task wlmPage(1)
    {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar4: !VPURT.Barrier) updates(%bar5: !VPURT.Barrier) wlmPage(2)
    {
        VPUIP.NNDMA <{port = 1 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar5: !VPURT.Barrier) updates(%bar6: !VPURT.Barrier) wlmPage(2)
    {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar5: !VPURT.Barrier) updates(%bar6: !VPURT.Barrier) wlmPage(2)
    {
        VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
            }>
            input(%buf0: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%buf2: memref<16x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%buf0: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%buf3: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%buf3: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants : {DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}
                        DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}} PPE : {}

    }

    VPURT.Task waits(%bar6: !VPURT.Barrier) updates(%bar7: !VPURT.Barrier) wlmPage(3)
    {
        VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
            }>
            input(%buf0: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%buf2: memref<16x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%buf0: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%buf3: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%buf3: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants : {DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}
                        DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}} PPE : {}

    }

    return %buf3: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // CHECK:   [[BAR0:%.+]] = VPURT.ConfigureBarrier<0> <{isStartBarrier, wlmPage = 0 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR1:%.+]] = VPURT.ConfigureBarrier<1> <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR2:%.+]] = VPURT.ConfigureBarrier<2> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR3:%.+]] = VPURT.ConfigureBarrier<3> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR4:%.+]] = VPURT.ConfigureBarrier<0> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR5:%.+]] = VPURT.ConfigureBarrier<1> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR6:%.+]] = VPURT.ConfigureBarrier<2> <{wlmPage = 3 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR7:%.+]] = VPURT.ConfigureBarrier<3> <{isFinalBarrier, wlmPage = 3 : i64}> -> !VPURT.Barrier

    // CHECK:   VPURT.Task updates([[BAR0]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.EnqueueDMA <{port = 0 : i64}>
    // CHECK-SAME: enqueue_dma_attr(<<SHAVE_ACT>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 0 : i64>)
    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 1 : i64}>
    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK:   VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 1 : i64}>
    // CHECK:   VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK:   VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.SW.Kernel
    // CHECK:   VPURT.Task waits([[BAR4]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK-NEXT: VPUIP.EnqueueDMA <{port = 0 : i64}>
    // CHECK-SAME: enqueue_dma_attr(<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 3 : i64>)
    // CHECK:   VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 1 : i64}>
    // CHECK:   VPURT.Task wlmPage(2) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK:   VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK:   VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK-NEXT: VPUIP.NCEClusterTask
    // CHECK:   VPURT.Task waits([[BAR6]] : !VPURT.Barrier) updates([[BAR7]] : !VPURT.Barrier) wlmPage(3) {
    // CHECK-NEXT: VPUIP.NCEClusterTask
}

// -----

// Test: start barrier (bar0) is consumed ONLY by a DPU task (not on P0 DDR queue).
// getBarrierEarliestConsumer(bar0) returns the DPU task index which is NOT on the P0 DDR
// queue used for enqueue DMA insertion. The fix looks for the earliest P0 DDR producer of
// bar0 and uses (producerIdx + 1) as the enqueue DMA insertion point, placing it correctly
// BEFORE the DPU consumer of bar0
//
//  DMA/DPU<HwFifo>[index]  bar<index>(PID)
//
// _______    DMA0[0]                    <- P0 DDR, produces bar0 (start barrier)
//              |
//            bar0(0) (start barrier)
//              |
//  Page0     DPU0[0]                    <- DPU is the ONLY (earliest) consumer of bar0,
//              |                           NOT on P0 DDR queue
//            bar1(1)
//              |
// _______   DMA0[1]
//              |
//            bar2(2)
//              |
//  Page1     DMA1[0]
//              |
//            bar3(3)
//              |
// _______   DMA0[2]
//              |
//            bar4(0)
//              |
//  Page2     DMA1[1]
//              |
//            bar5(1)
//           /      \
// _______  DMA0[3]  DPU0[1]
//            \      /
//             bar6(2)
//              |
//  Page3     DPU0[2]
//              |
//            bar7(3)
// _______

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @StartBarrierEarliestConsumerNotOnP0DDR() -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]> {
    %bar0 = VPURT.ConfigureBarrier<0> <{isStartBarrier, wlmPage = 0 : i64}> -> !VPURT.Barrier
    %bar1 = VPURT.ConfigureBarrier<1> <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    %bar2 = VPURT.ConfigureBarrier<2> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    %bar3 = VPURT.ConfigureBarrier<3> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    %bar4 = VPURT.ConfigureBarrier<0> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    %bar5 = VPURT.ConfigureBarrier<1> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    %bar6 = VPURT.ConfigureBarrier<2> <{wlmPage = 3 : i64}> -> !VPURT.Barrier
    %bar7 = VPURT.ConfigureBarrier<3> <{isFinalBarrier, wlmPage = 3 : i64}> -> !VPURT.Barrier

    %cst0 = const.Declare memref<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf1 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf2 = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %buf3 = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // DMA0[0]: P0 DDR produces start barrier bar0
    VPURT.Task updates(%bar0: !VPURT.Barrier) wlmPage(0) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    // DPU0[0]: DPU consumes bar0 - this is the earliest consumer, NOT on P0 DDR queue
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) wlmPage(0) {
        VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
            }>
            input(%buf0: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%buf2: memref<16x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%buf0: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%buf3: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%buf3: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants : {DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}
                        DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}} PPE : {}
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) wlmPage(0) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) wlmPage(1) {
        VPUIP.NNDMA <{port = 1 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar3: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) wlmPage(1) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar4: !VPURT.Barrier) updates(%bar5: !VPURT.Barrier) wlmPage(2) {
        VPUIP.NNDMA <{port = 1 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar5: !VPURT.Barrier) updates(%bar6: !VPURT.Barrier) wlmPage(2) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst0: memref<16x16x1x1xf16, {order = #NHWC}>) outputs(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    // DPU0[1]
    VPURT.Task waits(%bar5: !VPURT.Barrier) updates(%bar6: !VPURT.Barrier) wlmPage(2) {
        VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
            }>
            input(%buf0: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%buf2: memref<16x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%buf0: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%buf3: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%buf3: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants : {DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}
                        DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}} PPE : {}
    }

    // DPU0[2]
    VPURT.Task waits(%bar6: !VPURT.Barrier) updates(%bar7: !VPURT.Barrier) wlmPage(3) {
        VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
            }>
            input(%buf0: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%buf1: memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%buf2: memref<16x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%buf0: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%buf3: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%buf3: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
            variants : {DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}
                        DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}} PPE : {}
    }

    return %buf3: memref<1x16x8x32xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // CHECK:   [[BAR0:%.+]] = VPURT.ConfigureBarrier<0> <{isStartBarrier, wlmPage = 0 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR1:%.+]] = VPURT.ConfigureBarrier<1> <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR2:%.+]] = VPURT.ConfigureBarrier<2> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR3:%.+]] = VPURT.ConfigureBarrier<3> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR4:%.+]] = VPURT.ConfigureBarrier<0> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR5:%.+]] = VPURT.ConfigureBarrier<1> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR6:%.+]] = VPURT.ConfigureBarrier<2> <{wlmPage = 3 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR7:%.+]] = VPURT.ConfigureBarrier<3> <{isFinalBarrier, wlmPage = 3 : i64}> -> !VPURT.Barrier

    // DMA0[0] produces start barrier. EnqueueDMA for all DPU tasks is inserted right after it
    // (at P0 DDR producer + 1), before DPU0[0] which is bar0's earliest consumer.
    // CHECK:   VPURT.Task updates([[BAR0]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.EnqueueDMA <{port = 0 : i64}>
    // CHECK-SAME: enqueue_dma_attr(<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 1 : i64>)
    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NCEClusterTask
    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK:   VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.NNDMA <{port = 1 : i64}>
}
