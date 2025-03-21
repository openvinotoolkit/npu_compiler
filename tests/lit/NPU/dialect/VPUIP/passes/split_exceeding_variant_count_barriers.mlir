//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --split-exceeding-variant-count-barriers="max-variant-count=6" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExceedingProducersEqual
func.func @ExceedingProducersEqual() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //   0 1 2 3 4 5
    //        |
    //       bar0
    //        |
    //        6

    // multiple active barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    // active barrier consumer

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    //   0 1 2    3 4 5
    //     |        |
    //    bar0     bar1
    //     |        |
    //     6        6

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)

    // CHECK: VPURT.Task updates([[BAR1]] : !VPURT.Barrier)
    // CHECK: VPURT.Task updates([[BAR1]] : !VPURT.Barrier)
    // CHECK: VPURT.Task updates([[BAR1]] : !VPURT.Barrier)

    // CHECK: VPURT.Task waits([[BAR0]], [[BAR1]] : !VPURT.Barrier, !VPURT.Barrier)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExceedingProducersUnEqual
func.func @ExceedingProducersUnEqual() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //    0 1 2 3 4
    //        |
    //       bar0
    //        |
    //        6

    // multiple active barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    // active barrier consumer

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    //   0 1 2     3 4
    //     |        |
    //    bar0     bar1
    //     |        |
    //     6        6

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)

    // CHECK: VPURT.Task updates([[BAR1]] : !VPURT.Barrier)
    // CHECK: VPURT.Task updates([[BAR1]] : !VPURT.Barrier)

    // CHECK: VPURT.Task waits([[BAR0]], [[BAR1]] : !VPURT.Barrier, !VPURT.Barrier)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExceedingConsumersEqual
func.func @ExceedingConsumersEqual() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //        0
    //        |
    //       bar0
    //        |
    //   1 2 3 4 5 6

    // active barrier producer

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    // multiple active barrier consumers

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    //     0        0
    //     |        |
    //    bar0     bar1
    //     |        |
    //   1 2 3    4 5 6

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]], [[BAR1]] : !VPURT.Barrier, !VPURT.Barrier)

    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier)

    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExceedingConsumersUnEqual
func.func @ExceedingConsumersUnEqual() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //        0
    //        |
    //       bar0
    //        |
    //    1 2 3 4 6

    // active barrier producer

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    // multiple active barrier consumers

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    //     0        0
    //     |        |
    //    bar0     bar1
    //     |        |
    //   1 2 3     4 5

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]], [[BAR1]] : !VPURT.Barrier, !VPURT.Barrier)

    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier)

    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoChange
func.func @NoChange() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //      0 1 2
    //        |
    //       bar0
    //        |
    //      3 4 5

    // multiple active barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    // multiple active barrier consumers

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    //      0 1 2
    //        |
    //       bar0
    //        |
    //      3 4 5

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)

    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
}
