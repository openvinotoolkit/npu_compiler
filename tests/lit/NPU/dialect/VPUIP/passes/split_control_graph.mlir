//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt  --split-input-file --init-compiler="vpu-arch=%arch%" --split-control-graph="block-size=3" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// Note: 'idx' added since tasks can be reordered

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitControlGraph
func.func @SplitControlGraph() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
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

    // block 0:
    //              0
    //              |
    //            bar0
    //              |
    //              1
    //            /   \
    //          bar1   |
    //           |    bar2 (sync task wait bar)
    //            \   /
    //              2    (sync task)
    //              |
    //             bar3 (sync task update bar)
    //  block 1:  /   \
    //           |     |
    //           4     3
    //                 |
    //                bar4
    //                 |
    //                 5

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR4:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {idx = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]], [[BAR2]] : !VPURT.Barrier, !VPURT.Barrier) attributes {idx = 1 : i64}
    // CHECK: VPURT.Task waits([[BAR1]], [[BAR2]] : !VPURT.Barrier, !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) attributes {idx = 2 : i64, "sync-task"}
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) attributes {idx = 3 : i64}
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) attributes {idx = 4 : i64}
    // CHECK: VPURT.Task waits([[BAR4]] : !VPURT.Barrier) attributes {idx = 5 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitControlGraphWithImplictDmaDeps
func.func @SplitControlGraphWithImplictDmaDeps() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //  0       1
    //  |       |
    // bar0     2
    //  |       |
    //  |       3
    //  |       |
    //  4       5

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {idx = 0 : i64} {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task attributes {idx = 1 : i64} {
         VPUIP.NNDMA {port = 1 : i64}
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task attributes {idx = 2 : i64} {
         VPUIP.NNDMA {port = 1 : i64}
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task attributes {idx = 3 : i64} {
         VPUIP.NNDMA {port = 1 : i64}
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) attributes {idx = 4 : i64} {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task attributes {idx = 5 : i64} {
         VPUIP.NNDMA {port = 1 : i64}
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    // block 0:
    //           0     1
    //           |     |
    //          bar0   |
    //           |    bar1  (sync task wait bar)
    //            \   /
    //              2    (sync task)
    //              |
    //             bar2  (sync task update bar)
    //            /   \
    //  block 1: |     |
    //           4     3
    //                 |
    //                 5

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {idx = 0 : i64}
    // CHECK: VPURT.Task updates([[BAR1]] : !VPURT.Barrier) attributes {idx = 1 : i64}
    // CHECK: VPURT.Task waits([[BAR1]], [[BAR0]] : !VPURT.Barrier, !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {idx = 2 : i64, "sync-task"}
    // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) attributes {idx = 4 : i64}
    // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) attributes {idx = 3 : i64}
    // CHECK: VPURT.Task attributes {idx = 5 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitControlGraphSharedBar
func.func @SplitControlGraphSharedBar() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //  0  1  2  3
    //  |  |  |  |
    //   \ |  | /
    //     bar0
    //     |  |
    //     4  5

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {idx = 0 : i64} {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {idx = 1 : i64} {
         VPUIP.NNDMA {port = 1 : i64}
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {idx = 2 : i64} {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {idx = 3 : i64} {
         VPUIP.NNDMA {port = 1 : i64}
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) attributes {idx = 4 : i64} {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) attributes {idx = 5 : i64} {
         VPUIP.NNDMA {port = 1 : i64}
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    // block 0:
    //           0       1
    //            \     /
    //              bar0  (sync task wait bar)
    //               |
    //               2    (sync task)
    //               |
    //              bar1  (sync task update bar)
    //             / | \
    //  block 1:  /  |  \
    //           |   3   |
    //           |   |   |
    //           |  bar2 |
    //           |  / \  |
    //           | /   \ |
    //           4       5

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {idx = 0 : i64}
    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {idx = 1 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {idx = 2 : i64, "sync-task"}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {idx = 3 : i64}
    // CHECK: VPURT.Task waits([[BAR2]], [[BAR1]] : !VPURT.Barrier, !VPURT.Barrier) attributes {idx = 4 : i64}
    // CHECK: VPURT.Task waits([[BAR2]], [[BAR1]] : !VPURT.Barrier, !VPURT.Barrier) attributes {idx = 5 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitControlGraphLongProdRange
func.func @SplitControlGraphLongProdRanger() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //  0  1  2  3  4  5  6
    //  |  |  |  |  |  |  |
    //   \  \  \ | /  /  /
    //          bar0
    //           |
    //           7

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {idx = 0 : i64} {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {idx = 1 : i64} {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {idx = 2 : i64} {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) attributes {idx = 3 : i64} {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {idx = 4 : i64} {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {idx = 5 : i64} {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {idx = 6 : i64} {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) attributes {idx = 7 : i64} {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    // block 0:
    //           0       1
    //            \     /
    //              bar0  (sync task wait bar)
    //               |
    //               2    (sync task)
    //               |
    //           /- bar1  (sync task update bar)
    //         /  /     \
    //        |  3       4
    //         \  \     /
    //          \   bar2  (sync task wait bar)
    //            \  |
    //               5    (sync task)
    //               |
    //              bar3  (sync task update bar)
    //             / |
    //            |  6
    //            |  |
    //            \ bar4
    //             \ |
    //               7

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR4:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {idx = 0 : i64}
    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {idx = 1 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {idx = 2 : i64, "sync-task"}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {idx = 3 : i64}
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {idx = 4 : i64}
    // CHECK: VPURT.Task waits([[BAR2]], [[BAR1]] : !VPURT.Barrier, !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) attributes {idx = 5 : i64, "sync-task"}
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) attributes {idx = 6 : i64}
    // CHECK: VPURT.Task waits([[BAR4]], [[BAR3]] : !VPURT.Barrier, !VPURT.Barrier) attributes {idx = 7 : i64}
}
