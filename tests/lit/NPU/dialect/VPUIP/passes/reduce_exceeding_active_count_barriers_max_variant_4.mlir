//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --reduce-exceeding-active-count-barriers="num-barriers=2 max-variant-count=4" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// Note: 'idx' added since tasks can be reordered

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ParallelUpdateBarriers
func.func @ParallelUpdateBarriers() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //      0
    //      |
    //     bar0
    //      |
    //      1
    //     /  \
    //  bar1   bar2
    //   |      |
    //   2      3
    //   |      |
    //  bar3   bar4
    //   |      |
    //   4      5

    // multiple active barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {idx = 0 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1, %bar2: !VPURT.Barrier, !VPURT.Barrier) attributes {idx = 1 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) attributes {idx = 2 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar2 : !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) attributes {idx = 3 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar3 : !VPURT.Barrier) attributes {idx = 4 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar4 : !VPURT.Barrier) attributes {idx = 5 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    //      0
    //      |
    //     bar0
    //      |
    //      1
    //      |
    //     bar1
    //      |
    //      2
    //      |
    //     bar2
    //      |
    //      3
    //      |
    //     bar3
    //      /\
    //     4  5

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR4:%.*]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier}> -> !VPURT.Barrier
    // CHECK-NOT: VPURT.DeclareVirtual

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {idx = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {idx = 1 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {idx = 2 : i64}
    // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) attributes {idx = 3 : i64}
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) attributes {idx = 4 : i64}
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) attributes {idx = 5 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MergeParallelOutputPathsFromSharedBarrier
func.func @MergeParallelOutputPathsFromSharedBarrier() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //      0
    //      |
    //     bar0
    //      |
    //      1
    //      |
    //     bar1
    //    /   \
    //   2      3
    //   |      |
    //  bar2   bar3
    //   |      |
    //   4      5

    // multiple active barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {idx = 0 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) attributes {idx = 1 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) attributes {idx = 2 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) attributes {idx = 3 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar2 : !VPURT.Barrier) attributes {idx = 4 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar3 : !VPURT.Barrier) attributes {idx = 5 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    //      0
    //      |
    //     bar0
    //      |
    //      1
    //      |
    //     bar1
    //      |
    //      2
    //      |
    //     bar2
    //      |
    //      3
    //      |
    //     bar3
    //      /\
    //     4  5
    //

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR4:%.*]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier}> -> !VPURT.Barrier
    // CHECK-NOT: VPURT.DeclareVirtual

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {idx = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {idx = 1 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {idx = 2 : i64}
    // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) attributes {idx = 3 : i64}
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) attributes {idx = 4 : i64}
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) attributes {idx = 5 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @LinearizeBarrierWithMultipleConsumers
func.func @LinearizeBarrierWithMultipleConsumers() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //      0
    //      |
    //     bar0
    //      |
    //      1
    //      |
    //     bar1
    //    /   \
    //  2 3    4 5
    //   |      |
    //  bar2   bar3
    //   |      |
    //   6      7

    // multiple active barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {"idx" = 0 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) attributes {"idx" = 1 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) attributes {"idx" = 2 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) attributes {"idx" = 3 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) attributes {"idx" = 4 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) attributes {"idx" = 5 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar2 : !VPURT.Barrier) attributes {"idx" = 6 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar3 : !VPURT.Barrier) attributes {"idx" = 7 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    //      0
    //      |
    //     bar0
    //      |
    //      1
    //      |
    //     bar1
    //    /    \
    //    2    3
    //    \    /
    //     bar2
    //    /    \
    //    4    5
    //    \    /
    //     bar3
    //    /    \
    //    6    7

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR4:%.*]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier}> -> !VPURT.Barrier
    // CHECK-NOT: VPURT.DeclareVirtual

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {idx = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {idx = 1 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {idx = 2 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {idx = 3 : i64}
    // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) attributes {idx = 4 : i64}
    // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) attributes {idx = 5 : i64}
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) attributes {idx = 6 : i64}
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) attributes {idx = 7 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @LinearizeBarrierWithMultipleConsumersAndProducers
func.func @LinearizeBarrierWithMultipleConsumersAndProducers() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //      0
    //      |
    //     bar0
    //      |
    //      1
    //      |
    //     bar1
    //    /   \
    //  2 3    4 5
    //   |      |
    //  bar2   bar3
    //   |      |
    //  6 7    8 9

    // multiple active barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {"idx" = 0 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) attributes {"idx" = 1 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) attributes {"idx" = 2 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) attributes {"idx" = 3 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) attributes {"idx" = 4 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) attributes {"idx" = 5 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar2 : !VPURT.Barrier) attributes {"idx" = 6 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar2 : !VPURT.Barrier) attributes {"idx" = 7 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar3 : !VPURT.Barrier) attributes {"idx" = 8 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar3 : !VPURT.Barrier) attributes {"idx" = 9 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    //      0
    //      |
    //     bar0
    //      |
    //      1
    //      |
    //     bar1
    //    /    \
    //    2    3
    //    \    /
    //     bar2
    //    /    \
    //    4    5
    //    \    /
    //     bar3
    //    /    \
    //    6    7
    //    \    /
    //     bar4
    //    /    \
    //    8    9

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR4:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR5:%.*]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier}> -> !VPURT.Barrier
    // CHECK-NOT: VPURT.DeclareVirtual

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {idx = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {idx = 1 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {idx = 2 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {idx = 3 : i64}
    // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) attributes {idx = 4 : i64}
    // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) attributes {idx = 5 : i64}
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) attributes {idx = 6 : i64}
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) attributes {idx = 7 : i64}
    // CHECK: VPURT.Task waits([[BAR4]] : !VPURT.Barrier) attributes {idx = 8 : i64}
    // CHECK: VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier) attributes {idx = 9 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @LinearizeBarriersWithMultipleConsumersVariousOrder
func.func @LinearizeBarriersWithMultipleConsumersVariousOrder() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //      0
    //      |
    //     bar0
    //      |
    //      1
    //      |
    //     bar1
    //    /   \
    //  2 5    3 4
    //   |      |
    //  bar2   bar3
    //   |      |
    //   6      7

    // multiple active barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {"idx" = 0 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) attributes {"idx" = 1 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) attributes {"idx" = 2 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) attributes {"idx" = 3 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) attributes {"idx" = 4 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) attributes {"idx" = 5 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar2 : !VPURT.Barrier) attributes {"idx" = 6 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar3 : !VPURT.Barrier) attributes {"idx" = 7 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    //      0
    //      |
    //     bar0
    //      |
    //      1
    //      |
    //     bar1
    //    /    \
    //    2    3
    //    \    /
    //     bar2
    //      |
    //      4
    //      |
    //     bar3
    //      |
    //      5
    //      |
    //     bar4
    //    /    \
    //    6    7

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR4:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR5:%.*]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier}> -> !VPURT.Barrier
    // CHECK-NOT: VPURT.DeclareVirtual

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {idx = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {idx = 1 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {idx = 2 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {idx = 3 : i64}
    // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) attributes {idx = 4 : i64}
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) attributes {idx = 5 : i64}
    // CHECK: VPURT.Task waits([[BAR4]] : !VPURT.Barrier) attributes {idx = 6 : i64}
    // CHECK: VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier) attributes {idx = 7 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ParallelWaitBarriers
func.func @ParallelWaitBarriers() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //  0      1
    //  |      |
    // bar0   bar1
    //  |      |
    //  2      3
    //  |      |
    // bar2   bar3
    //    \  /
    //     4
    //     |
    //    bar4
    //     |
    //     5

    // multiple active barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {idx = 0 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar1: !VPURT.Barrier) attributes {idx = 1 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) attributes {idx = 2 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) attributes {idx = 3 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar2, %bar3 : !VPURT.Barrier, !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) attributes {idx = 4 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar4: !VPURT.Barrier) attributes {idx = 5 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    //      0
    //      |
    //     bar0
    //      |
    //      1
    //      |
    //     bar1
    //      |
    //      2
    //      |
    //     bar2
    //      |
    //      3
    //      |
    //     bar3
    //      |
    //      4
    //      |
    //     bar4
    //      |
    //      5

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR4:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR5:%.*]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier}> -> !VPURT.Barrier
    // CHECK-NOT: VPURT.DeclareVirtual

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {idx = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {idx = 1 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {idx = 2 : i64}
    // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) attributes {idx = 3 : i64}
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) attributes {idx = 4 : i64}
    // CHECK: VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier) attributes {idx = 5 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MergeParallelInputPathsFromSharedBarrier
func.func @MergeParallelInputPathsFromSharedBarrier() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //  0       1
    //  |       |
    // bar0    bar1
    //  |       |
    //  2       3
    //    \  /
    //    bar2
    //     |
    //     4
    //     |
    //    bar3
    //     |
    //     5

    // multiple active barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {idx = 0 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar1: !VPURT.Barrier) attributes {idx = 1 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) attributes {idx = 2 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) attributes {idx = 3 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar2 : !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) attributes {idx = 4 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar3: !VPURT.Barrier) attributes {idx = 5 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    //      0
    //      |
    //     bar0
    //      |
    //      1
    //      |
    //     bar1
    //    /    \
    //    2    3
    //    \    /
    //     bar2
    //      |
    //      4
    //      |
    //     bar3
    //      |
    //      5

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR4:%.*]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier}> -> !VPURT.Barrier
    // CHECK-NOT: VPURT.DeclareVirtual

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {idx = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {idx = 1 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {idx = 2 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {idx = 3 : i64}
    // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) attributes {idx = 4 : i64}
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) attributes {idx = 5 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @LinearizeBarriersWithMultiple
func.func @LinearizeBarriersWithMultipleProducers() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //  0 1  2 3
    //   |    |
    // bar0  bar1
    //   |    |
    //   4    5
    //    \  /
    //    bar2
    //     |
    //     6
    //     |
    //    bar3
    //     |
    //     7

    // multiple active barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {idx = 0 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {idx = 1 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar1: !VPURT.Barrier) attributes {idx = 2 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar1: !VPURT.Barrier) attributes {idx = 3 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) attributes {idx = 4 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) attributes {idx = 5 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar2 : !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) attributes {idx = 6 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar3: !VPURT.Barrier) attributes {idx = 7 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    //    0    1
    //    \    /
    //     bar0
    //    /    \
    //    2    3
    //    \    /
    //     bar1
    //    /    \
    //    4    5
    //    \    /
    //     bar2
    //      |
    //      6
    //      |
    //     bar3
    //      |
    //      7

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR4:%.*]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier}> -> !VPURT.Barrier
    // CHECK-NOT: VPURT.DeclareVirtual

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {idx = 0 : i64}
    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {idx = 1 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {idx = 2 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {idx = 3 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {idx = 4 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {idx = 5 : i64}
    // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) attributes {idx = 6 : i64}
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) attributes {idx = 7 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExtremeLinearization
func.func @ExtremeLinearization() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //  0 3  1 2
    //   |    |
    // bar0  bar1
    //   |    |
    //   4    5
    //    \  /
    //    bar2
    //     |
    //     6
    //     |
    //    bar3
    //     |
    //     7

    // multiple active barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {idx = 0 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar1: !VPURT.Barrier) attributes {idx = 1 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar1: !VPURT.Barrier) attributes {idx = 2 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {idx = 3 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) attributes {idx = 4 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) attributes {idx = 5 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar2 : !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) attributes {idx = 6 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar3: !VPURT.Barrier) attributes {idx = 7 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    //    0    1
    //    \    /
    //     bar0
    //      |
    //      2
    //      |
    //     bar1
    //      |
    //      3
    //      |
    //     bar2
    //    /    \
    //    4    5
    //    \    /
    //     bar3
    //      |
    //      6
    //      |
    //     bar4
    //      |
    //      7

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR4:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR5:%.*]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier}> -> !VPURT.Barrier
    // CHECK-NOT: VPURT.DeclareVirtual

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {idx = 0 : i64}
    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {idx = 1 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {idx = 2 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {idx = 3 : i64}
    // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) attributes {idx = 4 : i64}
    // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) attributes {idx = 5 : i64}
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) attributes {idx = 6 : i64}
    // CHECK: VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier) attributes {idx = 7 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TwoPaths
func.func @TwoPaths() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //  0       1
    //  |       |
    // bar0    bar1
    //  |       |
    //  2       3
    //  |       |
    // bar2    bar3
    //  |       |
    //  4       5

    // multiple active barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {idx = 0 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar1: !VPURT.Barrier) attributes {idx = 1 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) attributes {idx = 2 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) attributes {idx = 3 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar2 : !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) attributes {idx = 4 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar3: !VPURT.Barrier) attributes {idx = 5 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    //      0
    //      |
    //     bar0
    //      |
    //      1
    //      |
    //     bar1
    //    /    \
    //    2    3
    //    \    /
    //     bar2
    //      |
    //      4
    //      |
    //     bar3
    //      |
    //      5

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR4:%.*]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier}> -> !VPURT.Barrier
    // CHECK-NOT: VPURT.DeclareVirtual

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {idx = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {idx = 1 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {idx = 2 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {idx = 3 : i64}
    // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) attributes {idx = 4 : i64}
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) attributes {idx = 5 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ThreeOutputPathsFromSharedBarrier
func.func @ThreeOutputPathsFromSharedBarrier() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //         0
    //         |
    //        bar0
    //         |
    //         1
    //         |
    //        bar1
    //    /    |    \
    //   2     3     4
    //   |     |     |
    //  bar2  bar3  bar4
    //   |     |     |
    //   5     6     7

    // multiple active barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {idx = 0 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) attributes {idx = 1 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) attributes {idx = 2 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) attributes {idx = 3 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) attributes {idx = 4 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar2 : !VPURT.Barrier) attributes {idx = 5 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar3 : !VPURT.Barrier) attributes {idx = 6 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar4 : !VPURT.Barrier) attributes {idx = 7 : i64} {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    //      0
    //      |
    //     bar0
    //      |
    //      1
    //      |
    //     bar1
    //      |
    //      2
    //      |
    //     bar2
    //      |
    //      3
    //      |
    //     bar3
    //      |
    //      4
    //      |
    //     bar4
    //    /    \
    //    5    6
    //    \    /
    //     bar5
    //      |
    //      7

    // Note: better solution requires task re-ordering

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR4:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR5:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR6:%.*]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier}> -> !VPURT.Barrier
    // CHECK-NOT: VPURT.DeclareVirtual

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {idx = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {idx = 1 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {idx = 2 : i64}
    // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) attributes {idx = 3 : i64}
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) attributes {idx = 4 : i64}
    // CHECK: VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier) attributes {idx = 5 : i64}
    // CHECK: VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier) attributes {idx = 6 : i64}
    // CHECK: VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier) attributes {idx = 7 : i64}
}
