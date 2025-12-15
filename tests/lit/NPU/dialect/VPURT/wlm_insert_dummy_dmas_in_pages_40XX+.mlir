//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --wlm-insert-dummy-dmas-in-pages="num-barriers=4" %s | FileCheck %s
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

    // Simple subgraph with page split (barX(PageY)):
    //
    // _______      DMA0[0]
    //               |
    //              bar0(0)
    //               |
    //  Page0       DMA0[1]
    //               |
    //              bar1(0)
    //              /   \
    // _______ DMA0[2]  DPU[0]
    //            |  \   |
    //            |    \ |
    //  Page1  bar2(1)  bar3(1) <- Page 1 needs to have DMA1 task inserted
    //            |      |         as there is DMA1 task in Page2. Inserting this DMA
    // _______   DPU[2]  DPU[1]    will allow to enqueue DMA1[0] task at bootstrap, without
    //             \    /          dummyDMA, DMA1[0] would execute prematurely in Page0 which
    //              bar4(2)        will use same physical barriers as Page2
    //               |             Also DMA0 needs to be added as this DMA FIFO is used for
    //  Page2       DPU[3]         inserting enqueue DMAs in later passes
    //               |
    //              bar5(2)
    //               |
    // _______      DMA1[0]
    //               |
    //  Page3       bar6(3)
    // _______


    VPURT.Task updates(%bar0: !VPURT.Barrier) wlmPage(0)
    {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) wlmPage(0)
    {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2, %bar3: !VPURT.Barrier, !VPURT.Barrier) wlmPage(0)
    {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
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
        VPUIP.NNDMA {port = 1 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>


    // Simple subgraph with page split after inserting dummyDMAs:
    //
    // _______      DMA0[0]
    //               |
    //              bar0(0)
    //               |
    //  Page0       DMA0[1]
    //               |
    //              bar1(0)
    //              /   \
    // _______ DMA0[2]   DPU[0]
    //            |    \    |
    //            |      \  |
    //  Page1  bar2(1)     bar3(1) -------
    //            |       /     \         \
    // _______   DPU[2] DPU[1]  dummyDMA0  dummyDMA1
    //             \    /      /        __/
    //              bar4(2)   /      __/
    //               |       /    __/
    //  Page2       DPU[3]  /  __/
    //               |     /__/
    //              bar5(2)
    //               |
    // _______      DMA1[0]
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
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}

    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}

    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]], [[BAR3]] : !VPURT.Barrier, !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}

    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NCEClusterTask

    // CHECK:   VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.SyncDMA {port = 0 : i64}

    // CHECK:   VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.SyncDMA {port = 1 : i64}

    // CHECK:   VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.NCEClusterTask

    // CHECK:   VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.NCEClusterTask

    // CHECK:   VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK-NEXT: VPUIP.NCEClusterTask

    // CHECK:   VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
}

// -----

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
    // _______        DMA0[0]
    //                  |
    //                bar0(0)
    //                  |
    //  Page0         DMA0[1]
    //                  |
    //                bar1(0)
    //              /   |     \
    // _______ DMA0[3]  DPU[0]	DMA1[0] <- Task only updates Bar2 which makes this and next boundry task independent
    //            |  \  /	  /  /
    //            |   \/   /   /
    //            |   /\/     /
    //            |  //  \   /
    //  Page1  bar3(1)  bar2(1) <- Page 1 needs to have DMA1 task inserted
    //            |      |
    // _______   DPU[1]  DPU[2]
    //             \    /
    //              bar4(2)
    //               |
    //  Page2       DPU[3]
    //               |
    //              bar5(2)
    //               |
    // _______      DMA1[1]
    //               |
    //  Page3       bar6(3)
    // _______


    VPURT.Task updates(%bar0: !VPURT.Barrier) wlmPage(0)
    {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) wlmPage(0)
    {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2, %bar3: !VPURT.Barrier, !VPURT.Barrier) wlmPage(0)
    {
         VPUIP.NNDMA {port = 1 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2, %bar3: !VPURT.Barrier, !VPURT.Barrier) wlmPage(0)
    {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
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
        VPUIP.NNDMA {port = 1 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>


    // Simple subgraph with page split after inserting dummyDMAs:
    //
    // _______        DMA0[0]
    //                  \
    //                bar0(0)
    //                  \
    //  Page0         DMA0[1]
    //                  \
    //                bar1(0)
    //              /   |     \
    // _______ DMA0[3]  DMA1[0]	DPU[0]
    //            |\      / \   /
    //            | \   /   \  /
    //            |  \ /    \ /
    //            |  /\     \|
    //            | /  \    \
    //            |/    \   \
    //  Page1  bar2(1)     bar3(1) -------
    //            |       /     \         \
    // _______   DPU[2] DPU[1]  dummyDMA0  dummyDMA1
    //             \    /      /          /
    //              bar4(2)___/__________/
    //               |
    //  Page2       DPU[3]
    //               |
    //              bar5(2)
    //               |
    // _______      DMA1[0]
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
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}

    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}

    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]], [[BAR3]] : !VPURT.Barrier, !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}

    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]], [[BAR3]] : !VPURT.Barrier, !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}

    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NCEClusterTask

    // CHECK:   VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.SyncDMA {port = 0 : i64}

    // CHECK:   VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.SyncDMA {port = 1 : i64}

    // CHECK:   VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.NCEClusterTask

    // CHECK:   VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.NCEClusterTask

    // CHECK:   VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK-NEXT: VPUIP.NCEClusterTask

    // CHECK:   VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DmaAndDpuGraphWithGraphSplit() -> memref<1x16x8x32xf16,  #NHWC, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier <{isStartBarrier, wlmPage = 0 : i64}> -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    %bar5 = VPURT.DeclareVirtualBarrier <{wlmPage = 3 : i64}> -> !VPURT.Barrier
    %bar6 = VPURT.DeclareVirtualBarrier <{isFinalBarrier, wlmPage = 3 : i64}> -> !VPURT.Barrier

    // dummy buffer
    %cst0 = const.Declare memref<16x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %buf1 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %buf3 = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>

    // Simple subgraph with page split (barX(PageY)):
    //
    // _______      DMA0[0]
    //               |
    //              bar0(0)
    //               |
    //  Page0       DMA0[1]
    //               |
    //              bar1(0)
    //               |
    // _______      DMA0[2]
    //               |
    //  Page1       bar2(1)              <- Page1 needs to have dummy DMA1 inserted
    //               |                      This is single barrier page so before inserting dummy DMA1
    // _______      DMA0[3]  (sync-task)    page will be increased to have 2 barriers
    //               |
    //              bar3(2)
    //               |
    //  Page2       DMA0[4]              <- Page2 needs to have dummy DMA1 inserted in a way
    //               |                      to not create new boundary task to not violate
    //              bar4(2)                 control block split
    //               |
    // _______      DMA0[5]  (sync-task)
    //               |
    //              bar5(3)
    //               |
    //  Page3       DMA1[0]
    //               |
    //              bar6(3)
    // _______

    VPURT.Task updates(%bar0: !VPURT.Barrier) wlmPage(0)
    {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) wlmPage(0)
    {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) wlmPage(0)
    {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) wlmPage(1) attributes {"sync-task"}
    {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar3: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) wlmPage(2)
    {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar4: !VPURT.Barrier) updates(%bar5: !VPURT.Barrier) wlmPage(2)  attributes {"sync-task"}
    {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar5: !VPURT.Barrier) updates(%bar6: !VPURT.Barrier) wlmPage(3)
    {
        VPUIP.NNDMA {port = 1 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>


    // Simple subgraph with page split after inserting dummyDMAs:
    //
    // _______      DMA0[0]
    //               |
    //              bar0(0)
    //               |
    //  Page0       DMA0[1]
    //               |
    //              bar1(0)
    //               |
    // _______      DMA0[2]
    //               |
    //  Page1       bar2(1)
    //              /    \
    //         dummyDMA0 dummyDMA1
    //              \    /
    //              bar3(1)
    //               |
    // _______      DMA0[3]  (sync-task)
    //               |
    //              bar4(2)
    //              /    \
    //  Page2   DMA0[4]  dummyDMA1
    //              \    /
    //              bar5(2)
    //               |
    // _______      DMA0[5]  (sync-task)
    //               |
    //              bar6(3)
    //               |
    //  Page3       DMA1[0]
    //               |
    //              bar7(3)
    // _______


    // CHECK:   [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier <{isStartBarrier, wlmPage = 0 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR4:%.+]] = VPURT.DeclareVirtualBarrier <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR5:%.+]] = VPURT.DeclareVirtualBarrier <{wlmPage = 2 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR6:%.+]] = VPURT.DeclareVirtualBarrier <{wlmPage = 3 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR7:%.+]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier, wlmPage = 3 : i64}> -> !VPURT.Barrier

    // CHECK:   VPURT.Task updates([[BAR0]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}

    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}

    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) wlmPage(0) {
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}

    // CHECK:   VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.SyncDMA {port = 0 : i64}

    // CHECK:   VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) wlmPage(1) {
    // CHECK-NEXT: VPUIP.SyncDMA {port = 1 : i64}

    // CHECK:   VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) wlmPage(1) attributes {"sync-task"} {
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}

    // CHECK:   VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}

    // CHECK:   VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier) wlmPage(2) {
    // CHECK-NEXT: VPUIP.SyncDMA {port = 1 : i64}

    // CHECK:   VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier) wlmPage(2) attributes {"sync-task"} {
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}

    // CHECK:   VPURT.Task waits([[BAR6]] : !VPURT.Barrier) updates([[BAR7]] : !VPURT.Barrier) wlmPage(3) {
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
}
