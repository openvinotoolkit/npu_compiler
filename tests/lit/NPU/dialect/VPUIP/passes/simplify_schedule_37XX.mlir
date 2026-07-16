//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --simplify-schedule %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ShareParentChildBarriers
func.func @ShareParentChildBarriers() -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    //     DMA-0
    //       |
    //     Bar0
    //       |             Bar0
    //     DMA-1            |
    //       |     =>  DMA-0 DMA-1
    //     DMA-2            |
    //       |             Bar1
    //     Bar1

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        %0 = VPUIP.NNDMA <{port = 1 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
        %0 = VPUIP.NNDMA <{port = 0 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task updates(%bar1: !VPURT.Barrier) {
        %0 = VPUIP.NNDMA <{port = 0 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }


    return %buf1 : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    // CHECK: [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotShareParentChildBarriers
func.func @DoNotShareParentChildBarriers() -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        %0 = VPUIP.NNDMA <{port = 1 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
        %0 = VPUIP.NNDMA <{port = 0 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    // CHECK: [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
}
