//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --wlm-legalize-pages-for-barrier-dmas="num-barriers=4 barrier-fifo-depth=1" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DmaAndDpuGraph() -> memref<1x16x8x32xf16,  #NHWC, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier <{isStartBarrier, wlmPage = 0 : i64}> -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    %bar5 = VPURT.DeclareVirtualBarrier <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    %bar6 = VPURT.DeclareVirtualBarrier <{isFinalBarrier, wlmPage = 3 : i64}> -> !VPURT.Barrier

    // dummy buffer
    %cst0 = const.Declare memref<16x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
    %buf1 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %buf2 = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %buf3 = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>

    // Simple subgraph with page split (barX(PageY)):
    //
    // _______      DMA0
    //               |
    //              bar0(0)
    //               |
    //  Page0       DMA1
    //               |
    //              bar1(0)
    //              /   \
    // _______   DMA2   DPU0
    //            |  \   |
    //            |    \ |
    //  Page1  bar2(1)  bar3(1)
    //            |      |
    // _______   DPU2    DPU1
    //             \    /
    //              bar4(2)
    //               |
    //  Page2       DPU3
    //               |
    //              bar5(2)
    //               |
    // _______      DMA3
    //               |
    //  Page3       bar6(3)
    // _______


    VPURT.Task updates(%bar0: !VPURT.Barrier) wlmPage(0)
    {
         VPUIP.NNDMA inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) wlmPage(0)
    {
         VPUIP.NNDMA inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2, %bar3: !VPURT.Barrier, !VPURT.Barrier) wlmPage(0)
    {
         VPUIP.NNDMA inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) wlmPage(0)
    {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%buf2: memref<16x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) outputs(%buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
            variants : {DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}} PPE : {}

    }

    VPURT.Task waits(%bar3: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) wlmPage(1)
    {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%buf2: memref<16x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) outputs(%buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
            variants : {DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}} PPE : {}

    }

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) wlmPage(1)
    {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%buf2: memref<16x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) outputs(%buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
            variants : {DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}} PPE : {}

    }

    VPURT.Task waits(%bar4: !VPURT.Barrier) updates(%bar5: !VPURT.Barrier) wlmPage(2)
    {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%buf2: memref<16x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) outputs(%buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
            variants : {DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}} PPE : {}

    }

    VPURT.Task waits(%bar5: !VPURT.Barrier) updates(%bar6: !VPURT.Barrier) wlmPage(2)
    {
         VPUIP.NNDMA inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }


    return %buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>

    // After legalization (barX(PageY)):
    //
    // _______      DMA0
    //               |
    //              bar0(0)
    //               |
    //  Page0       DMA1
    //               |
    //              bar1(0)
    //              /   \
    // _______   DMA2   DPU0
    //            |  \ /   |
    //            |  / \   |
    //        bar2(1)   \  |
    //            |  \    \|
    //  Page1     | BarDMA |
    //            |     \  |
    //            |       bar3(1)
    //            |        |
    //            |      DPU1
    // _______   DPU2      |
    //             \      /
    //              bar4(2)
    //               |    \
    //  Page2       DPU3  BarDMA
    //               |    /
    //              bar5(2)
    //               |
    // _______      DMA3
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
    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]], [[BAR3]] : !VPURT.Barrier, !VPURT.Barrier) wlmPage(0) {
    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR3]], [[BAR2]] : !VPURT.Barrier, !VPURT.Barrier) wlmPage(0) {
    // CHECK:   VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.BarProgDMA {port = 0 : i64} inputs(%7 : memref<0x0x0x0xi32, @DDR>) outputs(%8 : memref<0x0x0x0xi32, @DDR>) physical_barrier_range(<0 : i64 to 1 : i64>)
    // CHECK:   VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK:   VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK:   VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK-NEXT: VPUIP.BarProgDMA {port = 0 : i64} inputs(%7 : memref<0x0x0x0xi32, @DDR>) outputs(%8 : memref<0x0x0x0xi32, @DDR>) physical_barrier_range(<2 : i64 to 3 : i64>)
    // CHECK:   VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK:   VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier) wlmPage(2) {
}
