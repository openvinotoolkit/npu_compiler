//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --wlm-insert-dummy-barriers-in-pages="num-barriers=4" %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DmaGraph() -> memref<16x16x1x1xf16,  #NHWC, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier <{isStartBarrier, wlmPage = 0 : i64}> -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier <{isFinalBarrier, wlmPage = 2 : i64}> -> !VPURT.Barrier

    // dummy buffer
    %cst0 = const.Declare memref<16x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>


    // Simple subgraph with page split (barX(PageY)):
    //
    // _______      DMA0[0]
    //               |
    //              bar0(0)  <- Page0 has only one barrier whereas page size = 2
    //               |          Insert dummy barrier parallel to bar0
    //  Page0       DMA0[1]
    //               |
    // _______      DMA0[2]
    //               |
    //              bar1(1)
    //               |
    //  Page1       DMA0[3]
    //               |
    //              bar2(1)
    //               |
    // _______      DMA0[4]
    //               |
    //  Page2       bar3(2)
    // _______


    VPURT.Task updates(%bar0: !VPURT.Barrier) wlmPage(0)
    {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf0: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) wlmPage(0)
    {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf0: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%bar1: !VPURT.Barrier) wlmPage(0)
    {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf0: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) wlmPage(1)
    {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf0: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) wlmPage(1)
    {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf0: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %buf0: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>


    // Simple subgraph with page split after inserting dummy barrier
    //
    // _______      DMA0[0]
    //               |      \
    //              bar0(0) barDummy(0)
    //               |      /
    //  Page0       DMA0[1]
    //               |
    // _______      DMA0[2]
    //               |
    //              bar1(1)
    //               |
    //  Page1       DMA0[3]
    //               |
    //              bar2(1)
    //               |
    // _______      DMA0[4]
    //               |
    //  Page2       bar3(2)
    // _______


    // CHECK:   [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier <{isStartBarrier, wlmPage = 0 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR_DUMMY:%.+]] = VPURT.DeclareVirtualBarrier <{wlmPage = 0 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    // CHECK:   [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier, wlmPage = 2 : i64}> -> !VPURT.Barrier

    // CHECK:   VPURT.Task updates([[BAR0]], [[BAR_DUMMY]] : !VPURT.Barrier, !VPURT.Barrier) wlmPage(0) {

    // CHECK:   VPURT.Task waits([[BAR0]], [[BAR_DUMMY]] : !VPURT.Barrier, !VPURT.Barrier) wlmPage(0) {

    // CHECK:   VPURT.Task updates([[BAR1]] : !VPURT.Barrier) wlmPage(0) {

    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) wlmPage(1) {

    // CHECK:   VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) wlmPage(1) {
}
