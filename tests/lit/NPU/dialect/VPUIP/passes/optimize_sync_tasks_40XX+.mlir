//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-sync-tasks %s | FileCheck %s
// REQUIRES: arch-NPU40XX
// Note: 'idx' added since tasks can be reordered

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MergeSyncTasks
func.func @MergeSyncTasks() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf2 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %buf3 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 0 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 1 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
        VPUIP.SyncDMA {port = 0 : i64} inputs(%buf2 : memref<0x0x0x0xi32, @DDR>) outputs(%buf3 : memref<0x0x0x0xi32, @DDR>) -> memref<0x0x0x0xi32, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
        VPUIP.SyncDMA {port = 0 : i64} inputs(%buf2 : memref<0x0x0x0xi32, @DDR>) outputs(%buf3 : memref<0x0x0x0xi32, @DDR>) -> memref<0x0x0x0xi32, @DDR>
    }

    VPURT.Task waits(%bar2: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 0 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar2: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 1 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}
    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.SyncDMA {port = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotRemoveSyncTaskIfItHasMultipleProducersAndConsumers
func.func @DoNotRemoveSyncTaskIfItHasMultipleProducersAndConsumers() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf2 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %buf3 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 0 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 1 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
        VPUIP.SyncDMA {port = 0 : i64} inputs(%buf2 : memref<0x0x0x0xi32, @DDR>) outputs(%buf3 : memref<0x0x0x0xi32, @DDR>) -> memref<0x0x0x0xi32, @DDR>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 0 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 1 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}
    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.SyncDMA
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RemoveRedundantSyncTaskBarDepAfter
func.func @RemoveRedundantSyncTaskBarDepAfter() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf2 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %buf3 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 0 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 1 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
        VPUIP.SyncDMA {port = 0 : i64} inputs(%buf2 : memref<0x0x0x0xi32, @DDR>) outputs(%buf3 : memref<0x0x0x0xi32, @DDR>) -> memref<0x0x0x0xi32, @DDR>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 0 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar2: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 1 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}
    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RemoveRedundantSyncTaskBarDepBefore
func.func @RemoveRedundantSyncTaskBarDepBefore() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf2 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %buf3 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 1 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 0 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
        VPUIP.SyncDMA {port = 0 : i64} inputs(%buf2 : memref<0x0x0x0xi32, @DDR>) outputs(%buf3 : memref<0x0x0x0xi32, @DDR>) -> memref<0x0x0x0xi32, @DDR>
    }

    VPURT.Task waits(%bar2: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 0 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar2: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 1 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RemoveRedundantSyncTaskFifoDepAfter
func.func @RemoveRedundantSyncTaskFifoDepAfter() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf2 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %buf3 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 0 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 1 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) {
        VPUIP.SyncDMA {port = 0 : i64} inputs(%buf2 : memref<0x0x0x0xi32, @DDR>) outputs(%buf3 : memref<0x0x0x0xi32, @DDR>) -> memref<0x0x0x0xi32, @DDR>
    }

    VPURT.Task updates(%bar1 : !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 0 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 1 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}
    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RemoveRedundantSyncTaskFifoDepBefore
func.func @RemoveRedundantSyncTaskFifoDepBefore() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf2 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %buf3 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 1 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 0 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar1 : !VPURT.Barrier) {
        VPUIP.SyncDMA {port = 0 : i64} inputs(%buf2 : memref<0x0x0x0xi32, @DDR>) outputs(%buf3 : memref<0x0x0x0xi32, @DDR>) -> memref<0x0x0x0xi32, @DDR>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 0 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 1 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RemoveRedundantSyncTaskIfFirst
func.func @RemoveRedundantSyncTaskIfFirst() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf2 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %buf3 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        VPUIP.SyncDMA {port = 0 : i64} inputs(%buf2 : memref<0x0x0x0xi32, @DDR>) outputs(%buf3 : memref<0x0x0x0xi32, @DDR>) -> memref<0x0x0x0xi32, @DDR>
    }
    VPURT.Task waits(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 1 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    // CHECK: VPURT.Task
    // CHECK-NOT:  VPUIP.SyncDMA
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RemoveRedundantSyncTaskIfFirstDoNotRemoveBar
func.func @RemoveRedundantSyncTaskIfFirstDoNotRemoveBar() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf2 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %buf3 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        VPUIP.SyncDMA {port = 0 : i64} inputs(%buf2 : memref<0x0x0x0xi32, @DDR>) outputs(%buf3 : memref<0x0x0x0xi32, @DDR>) -> memref<0x0x0x0xi32, @DDR>
    }
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 0 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }
    VPURT.Task waits(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 1 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK-NOT:  VPUIP.SyncDMA
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RemoveRedundantSyncTaskIfLast
func.func @RemoveRedundantSyncTaskIfLast() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf2 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %buf3 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 1 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) {
        VPUIP.SyncDMA {port = 0 : i64} inputs(%buf2 : memref<0x0x0x0xi32, @DDR>) outputs(%buf3 : memref<0x0x0x0xi32, @DDR>) -> memref<0x0x0x0xi32, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    // CHECK: VPURT.Task
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
    // CHECK-NOT:  VPUIP.SyncDMA

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RemoveRedundantSyncTaskFirstAndLast
func.func @RemoveRedundantSyncTaskFirstAndLast() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf2 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %buf3 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        VPUIP.SyncDMA {port = 0 : i64} inputs(%buf2 : memref<0x0x0x0xi32, @DDR>) outputs(%buf3 : memref<0x0x0x0xi32, @DDR>) -> memref<0x0x0x0xi32, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA  {port = 1 : i64} inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>) outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>) -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) {
        VPUIP.SyncDMA {port = 0 : i64} inputs(%buf2 : memref<0x0x0x0xi32, @DDR>) outputs(%buf3 : memref<0x0x0x0xi32, @DDR>) -> memref<0x0x0x0xi32, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    // CHECK: VPURT.Task
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
    // CHECK-NOT:  VPUIP.SyncDMA

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!CmxType = memref<1x64x32x32xf16, #NHWC, [@CMX_NN, 0]>

// CHECK-LABEL: @DoNotRemoveSyncTaskBarVariantLimit
func.func @DoNotRemoveSyncTaskBarVariantLimit() -> memref<1x64x32x32xf16, #NHWC, [@CMX_NN, 0]> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers
    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !CmxType
    %buf1 = VPURT.DeclareBuffer <CMX_NN> [0] <131072> -> !CmxType
    %buf2 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %buf3 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      VPUIP.NCEClusterTask {task_type = #VPUIP.nce_task_type<ELTWISE>} input(%buf0: !CmxType) weights(%buf0: !CmxType) parent_input(%buf0: !CmxType) parent_output(%buf1: !CmxType) outputs(%buf1: !CmxType) -> !CmxType variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
        VPUIP.SyncDMA {port = 0 : i64} inputs(%buf2 : memref<0x0x0x0xi32, @DDR>) outputs(%buf3 : memref<0x0x0x0xi32, @DDR>) -> memref<0x0x0x0xi32, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) {
      VPUIP.NCEClusterTask {task_type = #VPUIP.nce_task_type<ELTWISE>} input(%buf0: !CmxType) weights(%buf0: !CmxType) parent_input(%buf0: !CmxType) parent_output(%buf1: !CmxType) outputs(%buf1: !CmxType) -> !CmxType variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    return %buf1 :!CmxType

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-NOT: VPURT.DeclareVirtual

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NCEClusterTask
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NCEClusterTask
}
