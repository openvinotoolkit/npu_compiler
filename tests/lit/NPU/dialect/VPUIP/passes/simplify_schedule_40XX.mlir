//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --simplify-schedule="share-wait-and-update-barriers=true" %s | FileCheck %s
// REQUIRES: platform-NPU4000

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

// CHECK-LABEL: @OptimizeDoubleCycles
func.func @OptimizeDoubleCycles() -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    //            DMA-0                        DMA-0
    //              |                            |
    //             Bar0                         Bar0
    //              |   \                        |  \
    //             ...    \                     ...  DMA-1
    //              |       \                    |  /
    //             Bar1   DMA-1                 Bar1
    //           /  |       /                 /  |
    //         /   ...    /                 /   ...
    //       /      |   /         =>      /      |
    //     DMA-2   Bar2                 DMA-2   Bar3
    //       \      |                     \      |
    //         \   ...                      \   ...
    //           \  |                         \  |
    //             Bar3                         Bar2
    //              |                            |
    //            DMA-3                        DMA-3


    // multiple active barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 1 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 1 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 0 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 1 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 0 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 1 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar3: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 1 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    // CHECK: [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotOptimizeDoubleCycles
func.func @NotOptimizeDoubleCycles() -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    // multiple active barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 0 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 0 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 0 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 0 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 1 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 0 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar3: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 0 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    // CHECK: [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ShareParentChildBarriers
func.func @ShareParentChildBarriers() -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    //     DMA[00]-0
    //         |                DMA[00]-0
    //       Bar0                   |
    //         |                   Bar0
    //     DMA[00]-1                |
    //         |       =>  DMA[00]-1 DMA[00]-2
    //     DMA[00]-2                |
    //         |                   Bar1
    //       Bar1                   |
    //         |                DMA[00]-3
    //     DMA[00]-3

    // multiple active barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 0 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 0 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task updates(%bar1: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 0 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 0 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }


    return %buf1 : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    // CHECK: [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotShareParentChildBarriers
func.func @DoNotShareParentChildBarriers() -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    //     DMA[00]-0       DMA[01]-0
    //         |               |
    //        Bar0           Bar1
    //         |               |
    //     DMA[00]-0       DMA[01]-0

    // multiple active barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 0 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task updates(%bar1: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 1 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 0 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 1 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }


    return %buf1 : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    // CHECK: [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task updates([[BAR1]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
}
