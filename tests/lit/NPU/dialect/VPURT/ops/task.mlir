//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @PrintParseConfigureBarrier
func.func @PrintParseConfigureBarrier(%arg0: memref<1xf16>, %arg1: memref<1xf16>) -> memref<1xf16> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task {
        VPUIP.NNDMA {port = 0 : i64} inputs(%arg0: memref<1xf16>) outputs(%arg1: memref<1xf16>) -> memref<1xf16>
    }
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar0: !VPURT.Barrier) enqueueTarget(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%arg0: memref<1xf16>) outputs(%arg1: memref<1xf16>) -> memref<1xf16>
    }
    VPURT.Task wlmPage(1) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%arg0: memref<1xf16>) outputs(%arg1: memref<1xf16>) -> memref<1xf16>
    }
    VPURT.Task <{isTrailingSWLayer = true}> {
        VPUIP.NNDMA {port = 0 : i64} inputs(%arg0: memref<1xf16>) outputs(%arg1: memref<1xf16>) -> memref<1xf16>
    }
    VPURT.Task <{taskIndex = 1}> {
        VPUIP.NNDMA {port = 0 : i64} inputs(%arg0: memref<1xf16>) outputs(%arg1: memref<1xf16>) -> memref<1xf16>
    }
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar0: !VPURT.Barrier) enqueueTarget(%bar0: !VPURT.Barrier) wlmPage(1) <{isTrailingSWLayer = true, taskIndex = 1}> {
        VPUIP.NNDMA {port = 0 : i64} inputs(%arg0: memref<1xf16>) outputs(%arg1: memref<1xf16>) -> memref<1xf16>
    }

    return %arg1: memref<1xf16>

    // CHECK: [[BAR:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: VPURT.Task
    // CHECK: VPURT.Task waits([[BAR]] : !VPURT.Barrier) updates([[BAR]] : !VPURT.Barrier) enqueueTarget([[BAR]] : !VPURT.Barrier)
    // CHECK: VPURT.Task wlmPage(1)
    // CHECK: VPURT.Task <{isTrailingSWLayer = true}>
    // CHECK: VPURT.Task <{taskIndex = 1}>
    // CHECK: VPURT.Task waits([[BAR]] : !VPURT.Barrier) updates([[BAR]] : !VPURT.Barrier) enqueueTarget([[BAR]] : !VPURT.Barrier) wlmPage(1) <{isTrailingSWLayer = true, taskIndex = 1}>
}
