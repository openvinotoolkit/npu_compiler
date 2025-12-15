//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --check-wlm-page-split-constraints="num-barriers=4" %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DmaAndDpuGraph() -> memref<1x16x8x32xf16,  #NHWC, [@CMX_NN, 0]> {
    %bar0 = VPURT.ConfigureBarrier<0> <{isStartBarrier, wlmPage = 0 : i64}> -> !VPURT.Barrier
    %bar1 = VPURT.ConfigureBarrier<1> <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    %bar2 = VPURT.ConfigureBarrier<2> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    %bar3 = VPURT.ConfigureBarrier<3> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    %bar4 = VPURT.ConfigureBarrier<0> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    %bar5 = VPURT.ConfigureBarrier<1> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    %bar6 = VPURT.ConfigureBarrier<2> <{wlmPage = 3 : i64}> -> !VPURT.Barrier
    %bar7 = VPURT.ConfigureBarrier<3> <{isFinalBarrier, wlmPage = 3 : i64}> -> !VPURT.Barrier

    // dummy buffer
    %cst0 = const.Declare memref<16x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
    %buf1 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %buf2 = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %buf3 = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
    %buf4 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>

    // Simple subgraph with dummy ops:
    //  DMA/DPU<HwFifo>[index]
    //  bar<index>(PID)
    //
    // _______    FetchDma
    //               |
    //              DMA0[0]
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
    //             EnqueueDma
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

    VPURT.Task wlmPage(0)
    {
        VPUIP.FetchDMA {port = 0 : i64} inputs(%buf4 : memref<0x0x0x0xi32, @DDR>) outputs(%buf4 : memref<0x0x0x0xi32, @DDR>) fetch_dma(<<DPU>, tile = 0 : i64, list = 0 : i64, group = 0 : i64>) -> memref<0x0x0x0xi32, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) wlmPage(0)
    {
        VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) wlmPage(0)
    {
        VPUIP.NNDMA {port = 1 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) wlmPage(0)
    {
        VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) wlmPage(1)
    {
        VPUIP.NNDMA {port = 1 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar3 : !VPURT.Barrier) wlmPage(1) {
        VPUIP.EnqueueDMA {port = 0 : i64} inputs(%buf4 : memref<0x0x0x0xi32, @DDR>) outputs(%buf4 : memref<0x0x0x0xi32, @DDR>) enqueue_dma_attr(<<DPU>, tile = 0 : i64, list = 0 : i64, startTask = 0 : i64, endTask = 3 : i64>) -> memref<0x0x0x0xi32, @DDR>
    }

    VPURT.Task waits(%bar3: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) wlmPage(1)
    {
        VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar4: !VPURT.Barrier) updates(%bar5: !VPURT.Barrier) wlmPage(2)
    {
        VPUIP.NNDMA {port = 1 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar5: !VPURT.Barrier) updates(%bar6: !VPURT.Barrier) wlmPage(2)
    {
        VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar5: !VPURT.Barrier) updates(%bar6: !VPURT.Barrier) wlmPage(2)
    {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%buf2: memref<16x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) outputs(%buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
            variants : {DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}
                        DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}} PPE : {}

    }

    VPURT.Task waits(%bar6: !VPURT.Barrier) updates(%bar7: !VPURT.Barrier) wlmPage(3)
    {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%buf2: memref<16x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) outputs(%buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
            variants : {DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}
                        DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}} PPE : {}

    }

    return %buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:   [[BAR0:%.+]] = VPURT.ConfigureBarrier<0> <{isStartBarrier, wlmPage = 0 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR1:%.+]] = VPURT.ConfigureBarrier<1> <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR2:%.+]] = VPURT.ConfigureBarrier<2> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR3:%.+]] = VPURT.ConfigureBarrier<3> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR4:%.+]] = VPURT.ConfigureBarrier<0> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR5:%.+]] = VPURT.ConfigureBarrier<1> <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR6:%.+]] = VPURT.ConfigureBarrier<2> <{wlmPage = 3 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR7:%.+]] = VPURT.ConfigureBarrier<3> <{isFinalBarrier, wlmPage = 3 : i64}> -> !VPURT.Barrier

    // CHECK:   VPURT.Task wlmPage(0) {
    // CHECK-NEXT: VPUIP.FetchDMA {port = 0 : i64}
    // CHECK:   VPURT.Task updates([[BAR0]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}
    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}
    // CHECK:   VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
    // CHECK:   VPURT.Task waits([[BAR3]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.EnqueueDMA {port = 0 : i64}
    // CHECK:   VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}
    // CHECK:   VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
    // CHECK:   VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}
    // CHECK:   VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK-NEXT: VPUIP.NCEClusterTask
    // CHECK:   VPURT.Task waits([[BAR6]] : !VPURT.Barrier) updates([[BAR7]] : !VPURT.Barrier) wlmPage(3) {
    // CHECK-NEXT: VPUIP.NCEClusterTask

}
