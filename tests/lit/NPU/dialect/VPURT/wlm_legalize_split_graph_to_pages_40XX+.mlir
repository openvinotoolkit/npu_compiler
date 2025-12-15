//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --wlm-legalize-split-graph-to-pages="num-barriers=4" %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

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

    // Simple subgraph with dummy ops:
    // _______      DMA0
    //               |
    //              bar0(0)
    //               |
    //  Page0       DMA1
    //               |
    //              bar1(0)
    //              /   \
    // _______   DMA2   DPU0
    //            |      |
    //            |      |
    //  Page1  bar2(1)  bar3(1)
    //            |      |
    // _______   DPU1    DPU2
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

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) wlmPage(0)
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

    // After page split (barX(PageY)):
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
    // _______   DPU2    DPU1         On Page1 boundary tasks it is guaranteed that all
    //             \    /             Page0 barriers have been consumed and Page2 can reuse
    //              bar4(2)           physical barriers from Page0
    //               |
    //  Page2       DPU3
    //               |
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
    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK:   VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK:   VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK:   VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK:   VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier) wlmPage(2) {
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

!type_DDR = memref<1x1x32x375xf16, @DDR>
!type_CMX = memref<1x1x32x375xf16, [@CMX_NN, 0]>

func.func @DmaGraphs() -> (!type_CMX, !type_CMX, !type_CMX, !type_CMX, !type_CMX, !type_CMX, !type_CMX) {
    %bar0 = VPURT.DeclareVirtualBarrier <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier <{wlmPage = 2 : i64}> -> !VPURT.Barrier

    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <576000> -> !type_CMX
    %buf1 = VPURT.DeclareBuffer <CMX_NN> [0] <600000> -> !type_CMX
    %buf2 = VPURT.DeclareBuffer <CMX_NN> [0] <624000> -> !type_CMX
    %buf3 = VPURT.DeclareBuffer <CMX_NN> [0] <648000> -> !type_CMX
    %buf4 = VPURT.DeclareBuffer <CMX_NN> [0] <672000> -> !type_CMX
    %buf5 = VPURT.DeclareBuffer <CMX_NN> [0] <696000> -> !type_CMX
    %buf6 = VPURT.DeclareBuffer <CMX_NN> [0] <720000> -> !type_CMX
    %buf7 = VPURT.DeclareBuffer <CMX_NN> [0] <744000> -> !type_CMX
    %buf8 = VPURT.DeclareBuffer <CMX_NN> [0] <768000> -> !type_CMX
    %buf9 = VPURT.DeclareBuffer <CMX_NN> [0] <792000> -> !type_CMX
    %buf10 = VPURT.DeclareBuffer <CMX_NN> [0] <1392000> -> !type_CMX
    %buf11 = VPURT.DeclareBuffer <CMX_NN> [0] <1416000> -> !type_CMX
    %buf12 = VPURT.DeclareBuffer <CMX_NN> [0] <1440000> -> !type_CMX
    %buf13 = VPURT.DeclareBuffer <CMX_NN> [0] <1464000> -> !type_CMX
    %buf14 = VPURT.DeclareBuffer <NetworkInput> [0] <4500> -> !type_DDR
    %buf15 = VPURT.DeclareBuffer <NetworkInput> [0] <388500> -> !type_DDR
    %buf16 = VPURT.DeclareBuffer <NetworkInput> [0] <772500> -> !type_DDR
    %buf17 = VPURT.DeclareBuffer <NetworkInput> [0] <1156500> -> !type_DDR
    %buf18 = VPURT.DeclareBuffer <NetworkInput> [0] <6000> -> !type_DDR
    %buf19 = VPURT.DeclareBuffer <NetworkInput> [0] <394500> -> !type_DDR
    %buf20 = VPURT.DeclareBuffer <NetworkInput> [0] <778500> -> !type_DDR

    // Simple subgraph with DMA ops:
    // _______      DMA0        DMA1        DMA2        DMA3        DMA4        DMA5        DMA6
    //               |                     /                       /                          /
    //              bar0(0)          bar1(0)                      /                          /
    //  Page0        |               /                           /                          /
    //               |              /                           /                          /
    //              DMA7        DMA8        DMA9               /                          /
    // _______                                                /                          /
    //                                                       /                          /
    //                                                    bar2(1)                      /
    //  Page1                                              /                          /
    //                                                  DMA10       DMA11            /
    // _______                                                                      /
    //                                                                             /
    //                                                                          bar3(2)
    //                                                                           /
    //  Page2                                                                   DMA12       DMA13
    //                                                                                       |
    //                                                                                      bar4(2)
    // _______

    VPURT.Task updates(%bar0 : !VPURT.Barrier) wlmPage(0)
    {
        VPUIP.NNDMA {port = 0 : i64} inputs(%buf14 : !type_DDR) outputs(%buf0 : !type_CMX) -> !type_CMX
    }

    VPURT.Task wlmPage(0)
    {
        VPUIP.NNDMA {port = 0 : i64} inputs(%buf15 : !type_DDR) outputs(%buf2 : !type_CMX) -> !type_CMX
    }

    VPURT.Task updates(%bar1 : !VPURT.Barrier) wlmPage(0)
    {
        VPUIP.NNDMA {port = 0 : i64} inputs(%buf16 : !type_DDR) outputs(%buf4 : !type_CMX) -> !type_CMX
    }

    VPURT.Task wlmPage(0)
    {
        VPUIP.NNDMA {port = 0 : i64} inputs(%buf17 : !type_DDR) outputs(%buf6 : !type_CMX) -> !type_CMX
    }

    VPURT.Task updates(%bar2 : !VPURT.Barrier) wlmPage(0)
    {
        VPUIP.NNDMA {port = 0 : i64} inputs(%buf18 : !type_DDR) outputs(%buf8 : !type_CMX) -> !type_CMX
    }

    VPURT.Task wlmPage(0)
    {
        VPUIP.NNDMA {port = 0 : i64} inputs(%buf19 : !type_DDR) outputs(%buf10 : !type_CMX) -> !type_CMX
    }

    VPURT.Task updates(%bar3 : !VPURT.Barrier) wlmPage(0)
    {
        VPUIP.NNDMA {port = 0 : i64} inputs(%buf20 : !type_DDR) outputs(%buf12 : !type_CMX) -> !type_CMX
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) wlmPage(0)
    {
        VPUIP.NNDMA {port = 0 : i64} inputs(%buf0 : !type_CMX) outputs(%buf1 : !type_CMX) -> !type_CMX
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) wlmPage(0)
    {
        VPUIP.NNDMA {port = 0 : i64} inputs(%buf2 : !type_CMX) outputs(%buf3 : !type_CMX) -> !type_CMX
    }

    VPURT.Task wlmPage(0)
    {
        VPUIP.NNDMA {port = 0 : i64} inputs(%buf4 : !type_CMX) outputs(%buf5 : !type_CMX) -> !type_CMX
    }

    VPURT.Task waits(%bar2 : !VPURT.Barrier) wlmPage(1)
    {
        VPUIP.NNDMA {port = 0 : i64} inputs(%buf6 : !type_CMX) outputs(%buf7 : !type_CMX) -> !type_CMX
    }

    VPURT.Task wlmPage(1)
    {
        VPUIP.NNDMA {port = 0 : i64} inputs(%buf8 : !type_CMX) outputs(%buf9 : !type_CMX) -> !type_CMX
    }

    VPURT.Task waits(%bar3 : !VPURT.Barrier) wlmPage(2)
    {
        VPUIP.NNDMA {port = 0 : i64} inputs(%buf10 : !type_CMX) outputs(%buf11 : !type_CMX) -> !type_CMX
    }

    VPURT.Task updates(%bar4: !VPURT.Barrier) wlmPage(2)
    {
        VPUIP.NNDMA {port = 0 : i64} inputs(%buf12 : !type_CMX) outputs(%buf13 : !type_CMX) -> !type_CMX
    }

    return %buf1, %buf3, %buf5, %buf7, %buf9, %buf11, %buf13 : !type_CMX, !type_CMX, !type_CMX, !type_CMX, !type_CMX, !type_CMX, !type_CMX

    // CHECK:   [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR4:%.+]] = VPURT.DeclareVirtualBarrier <{wlmPage = 2 : i64}> -> !VPURT.Barrier

    // CHECK:   VPURT.Task updates([[BAR0]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK:   VPURT.Task wlmPage(0) {
    // CHECK:   VPURT.Task updates([[BAR1]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK:   VPURT.Task wlmPage(0) {
    // CHECK:   VPURT.Task updates([[BAR2]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK:   VPURT.Task wlmPage(0) {
    // CHECK:   VPURT.Task updates([[BAR2]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK:   VPURT.Task wlmPage(0) {
    // CHECK:   VPURT.Task waits([[BAR2]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK:   VPURT.Task updates([[BAR3]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK:   VPURT.Task waits([[BAR3]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK:   VPURT.Task updates([[BAR4]] : !VPURT.Barrier) wlmPage(2) {
}
