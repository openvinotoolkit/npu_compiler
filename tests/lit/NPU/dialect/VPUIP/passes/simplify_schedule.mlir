//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --simplify-schedule %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @tasksWithoutDeclareBuffer
func.func @tasksWithoutDeclareBuffer(%arg0: memref<1x1x1x1xf16>, %arg1: memref<1x1x1x1xf16>) -> memref<1x1x1x1xf16> {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task updates(%0 : !VPURT.Barrier){
          VPUIP.NNDMA {port = 0 : i64}
              inputs(%arg0: memref<1x1x1x1xf16>)
              outputs(%arg1: memref<1x1x1x1xf16>)
              -> memref<1x1x1x1xf16>
    }
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task waits(%0: !VPURT.Barrier) updates(%1: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%arg0: memref<1x1x1x1xf16>)
            outputs(%arg1: memref<1x1x1x1xf16>)
            -> memref<1x1x1x1xf16>
    }
    VPURT.Task waits(%1: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%arg0: memref<1x1x1x1xf16>)
            outputs(%arg1: memref<1x1x1x1xf16>)
            -> memref<1x1x1x1xf16>
    }
    return %arg1 : memref<1x1x1x1xf16>
    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
}

// -----

// CHECK-LABEL: @tasksWithoutDeclareBufferAndVirtualBarrier
func.func @tasksWithoutDeclareBufferAndVirtualBarrier(%arg0: memref<1x1x1x1xf16>, %arg1: memref<1x1x1x1xf16>) -> memref<1x1x1x1xf16> {
    VPURT.Task {
          VPUIP.NNDMA {port = 0 : i64}
              inputs(%arg0: memref<1x1x1x1xf16>)
              outputs(%arg1: memref<1x1x1x1xf16>)
              -> memref<1x1x1x1xf16>
    }
    VPURT.Task {
         VPUIP.NNDMA {port = 1 : i64}
            inputs(%arg0: memref<1x1x1x1xf16>)
            outputs(%arg1: memref<1x1x1x1xf16>)
            -> memref<1x1x1x1xf16>
    }
    return %arg1 : memref<1x1x1x1xf16>
    // CHECK: VPURT.Task
    // CHECK: VPURT.Task
}
